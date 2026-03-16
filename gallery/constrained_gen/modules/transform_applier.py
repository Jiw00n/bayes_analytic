"""
transform_applier — TransformApplier: SymbolicState에 transform step을 적용하는 로직.
"""
import tvm
from tvm import tir
from tvm.auto_scheduler.loop_state import StateObject

from .sym_types import (
    SymExpr, SymIter, SymStage, eval_sym_extent,
    CA_ROOT, CA_INLINED, CA_ITER,
)


class TransformApplier:
    """
    SymbolicState에 transform steps를 순차 적용하는 로직.
    SymbolicState의 내부 데이터(stages, sym_map, 메타데이터)를 직접 조작.
    """

    def __init__(self, sym_state):
        """적용 대상 SymbolicState를 받아 보관한다."""
        self.s = sym_state

    @staticmethod
    def _clamp_positive_split_extent(sym_ext, tosplit_extent):
        """Split extent clamp for strictly-positive extents.

        Split factors and loop extents in this path are positive. When either
        side is the concrete constant 1, the clamped extent is also exactly 1,
        so avoid materializing symbolic forms like min(sp_i_j,1).
        """
        if sym_ext is None or tosplit_extent is None:
            return sym_ext if tosplit_extent is None else tosplit_extent

        sym_val = sym_ext.val if isinstance(sym_ext, SymExpr) else sym_ext
        clamp_val = tosplit_extent.val if isinstance(tosplit_extent, SymExpr) else tosplit_extent
        if sym_val == 1 or clamp_val == 1:
            return SymExpr(1)
        return SymExpr.min(sym_ext, tosplit_extent)

    def apply_steps(self, state):
        """모든 transform steps를 순차 적용."""
        self.s._state = state
        steps = state.transform_steps
        for i, step in enumerate(steps):
            tk = step.type_key.split(".")[-1]
            if tk == "AnnotationStep":
                self._apply_annotation(step)
            elif tk == "FuseStep":
                self._apply_fuse(step, i)
            elif tk == "PragmaStep":
                self._apply_pragma(step, i)
            elif tk == "ReorderStep":
                self._apply_reorder(step)
            elif tk == "SplitStep":
                self._apply_split(step, i)
            elif tk == "FollowSplitStep":
                self._apply_follow_split(step, steps, i)
            elif tk == "FollowFusedSplitStep":
                self._apply_follow_fused_split(step, steps, i)
            elif tk == "StorageAlignStep":
                self._apply_storage_align(step)
            elif tk == "ComputeAtStep":
                self._apply_compute_at(step)
            elif tk == "ComputeInlineStep":
                self._apply_compute_inline(step)
            elif tk == "ComputeRootStep":
                self._apply_compute_root(step)
            elif tk == "CacheReadStep":
                self._apply_cache_read(step, state, i)
            elif tk == "CacheWriteStep":
                self._apply_cache_write(step, state, i)
            else:
                raise NotImplementedError(f"Unhandled transform step at index {i}: {tk}")

        self._infer_bound_final(state)

    # ─── Lazy extent 복원 ───
    @staticmethod
    def _clone_real_stage(real_stage):
        new_iters = []
        for it in real_stage.iters:
            ext = int(it.range.extent) if it.range is not None else None
            new_iters.append(
                SymIter(it.name, SymExpr(ext) if ext is not None else None, annotation=0, iter_kind=0)
            )
        return SymStage(
            real_stage.op.name,
            'compute',
            new_iters,
            dtype=str(real_stage.op.output(0).dtype) if hasattr(real_stage.op, 'output') else "float32",
        )

    @staticmethod
    def _product_of_defined_iters(stage):
        product = SymExpr(1)
        for it in stage.iters:
            if it.extent is None:
                return None
            product = SymExpr.mul(product, it.extent)
        return product

    def _get_cache_read_restore_ctx(self, stage_id):
        s = self.s
        if stage_id not in s._cache_read_consumer:
            return None, None, None

        cr_sym_candidates = self._get_consumer_split_sym_products(stage_id)
        cr_stencil = s._cache_read_stencil_info.get(stage_id)
        cr_ordered_splits = None
        if cr_stencil:
            consumer_sid = s._cache_read_consumer[stage_id]
            ordered = [
                (si, prod)
                for (sid, si), prod in s._split_sym_products.items()
                if sid == consumer_sid
            ]
            ordered.sort(key=lambda x: x[0])
            cr_ordered_splits = [
                (SymExpr(prod.val), eval_sym_extent(prod, s.sym_map))
                for si, prod in ordered
            ]
        return cr_sym_candidates, cr_stencil, cr_ordered_splits

    def _get_safe_saved_extent(self, stage_id, iter_id, real_ext):
        del real_ext
        saved = self.s._ca_saved_extents.get((stage_id, iter_id))
        if saved is None:
            return None
        return SymExpr(saved.val)

    def _recover_iter_extent(self, stage_id, iter_id, real_ext,
                             cr_sym_candidates=None, cr_stencil=None, cr_ordered_splits=None):
        stage = self.s.stages[stage_id]
        saved_present = (stage_id, iter_id) in self.s._ca_saved_extents

        if cr_sym_candidates is not None:
            if cr_stencil and cr_ordered_splits and iter_id in cr_stencil:
                pattern_kind, pattern_value, sp_order, rd_order = cr_stencil[iter_id]
                if pattern_kind == "direct":
                    order = sp_order if sp_order is not None else rd_order
                    if order is not None and order < len(cr_ordered_splits):
                        sym_expr, eval_val = cr_ordered_splits[order]
                        for ci_idx, (ev, se) in enumerate(cr_sym_candidates):
                            if ev == eval_val and se.val == sym_expr.val:
                                cr_sym_candidates.pop(ci_idx)
                                break
                        return sym_expr
                elif pattern_kind == "linear" and sp_order is not None and rd_order is not None:
                    stride = pattern_value
                    sp_sym, sp_eval = cr_ordered_splits[sp_order]
                    rd_sym, rd_eval = cr_ordered_splits[rd_order]
                    predicted = (sp_eval - 1) * stride + rd_eval
                    if predicted == real_ext:
                        return SymExpr(f"({sp_sym.val} - 1)*{stride} + {rd_sym.val}")
                elif pattern_kind == "grouped" and sp_order is not None and rd_order is not None:
                    block = pattern_value
                    sp_sym, sp_eval = cr_ordered_splits[sp_order]
                    rd_sym, rd_eval = cr_ordered_splits[rd_order]
                    predicted = (((sp_eval + block - 1) // block) - 1) * block + rd_eval
                    if predicted == real_ext:
                        return SymExpr(f"(ceil({sp_sym.val}/({block})) - 1)*{block} + {rd_sym.val}")

            matched_sym = self._match_cr_extent(real_ext, cr_sym_candidates)
            if matched_sym is not None:
                cr_sym_candidates.remove((real_ext, matched_sym))
                return matched_sym

        saved = self._get_safe_saved_extent(stage_id, iter_id, real_ext)
        if saved is not None:
            return saved

        if not saved_present:
            ca_match = self._match_compute_at_inner_extent(stage_id, iter_id, real_ext)
            if ca_match is not None:
                return ca_match

        if stage.compute_at == CA_ITER and real_ext == 1:
            return SymExpr(1)

        return SymExpr(real_ext)

    def _restore_stage_extents_if_needed(self, stage_id, step_idx):
        s = self.s
        stage = s.stages[stage_id]
        has_none = any(it.extent is None for it in stage.iters)
        if not has_none:
            return

        ps = tvm._ffi.get_global_func("auto_scheduler.ReplayStepsPartial")(
            s.compute_dag, s._state, step_idx)
        bounded = s.compute_dag.infer_bound_from_state(ps)

        if stage_id >= len(bounded.stages):
            return

        cr_sym_candidates, cr_stencil, cr_ordered_splits = self._get_cache_read_restore_ctx(stage_id)

        real_stage = bounded.stages[stage_id]
        for iid in range(len(stage.iters)):
            if stage.iters[iid].extent is None and iid < len(real_stage.iters):
                real_it = real_stage.iters[iid]
                if real_it.range is not None:
                    real_ext = int(real_it.range.extent)
                    stage.iters[iid].extent = self._recover_iter_extent(
                        stage_id, iid, real_ext,
                        cr_sym_candidates=cr_sym_candidates,
                        cr_stencil=cr_stencil,
                        cr_ordered_splits=cr_ordered_splits,
                    )

    def _get_consumer_split_sym_products(self, cache_read_stage_id):
        s = self.s
        consumer_sid = s._cache_read_consumer.get(cache_read_stage_id)
        if consumer_sid is None:
            return None
        candidates = []
        for (sid, step_idx), sym_prod in s._split_sym_products.items():
            if sid == consumer_sid:
                eval_val = eval_sym_extent(sym_prod, s.sym_map)
                if isinstance(eval_val, int):
                    candidates.append((eval_val, SymExpr(sym_prod.val)))
        return candidates

    @staticmethod
    def _match_cr_extent(real_ext, candidates):
        for eval_val, sym_expr in candidates:
            if eval_val == real_ext:
                return sym_expr
        return None

    @staticmethod
    def _iter_base_name(name):
        name = str(name)
        if "@@" in name:
            return name.split("@@", 1)[0]
        if "@" in name:
            return name.split("@", 1)[0]
        if "." in name:
            return name.split(".", 1)[0]
        return name

    def _match_compute_at_inner_extent(self, sid, iid, real_ext):
        """compute_at된 stage의 iter에 대해, target stage의 inner iter 중
        eval(extent)==real_ext 인 non-concrete symbolic extent를 찾아 반환."""
        s = self.s
        stage = s.stages[sid]
        if stage.compute_at != CA_ITER or stage.attach_stage_id is None:
            return None
        target_sid = stage.attach_stage_id
        target_iid = stage.attach_iter_id
        target_stage = s.stages[target_sid]
        iter_base = self._iter_base_name(stage.iters[iid].name)
        for tiid in range(target_iid + 1, len(target_stage.iters)):
            t_it = target_stage.iters[tiid]
            if t_it.extent is None or t_it.extent.is_concrete:
                continue
            target_base = self._iter_base_name(t_it.name)
            if target_base != iter_base:
                continue
            t_eval = eval_sym_extent(t_it.extent, s.sym_map)
            if isinstance(t_eval, int) and t_eval == real_ext:
                return SymExpr(t_it.extent.val)
        return None

    def _infer_bound_final(self, state):
        s = self.s
        state_obj = state if isinstance(state, StateObject) else state.state_object
        bounded = s.compute_dag.infer_bound_from_state(state_obj)

        for sid in range(len(s.stages)):
            sym_stage = s.stages[sid]
            if sid >= len(bounded.stages):
                continue
            real_stage = bounded.stages[sid]

            cr_sym_candidates, cr_stencil, cr_ordered_splits = self._get_cache_read_restore_ctx(sid)

            for iid in range(len(sym_stage.iters)):
                sym_it = sym_stage.iters[iid]
                if sym_it.extent is None and iid < len(real_stage.iters):
                    real_it = real_stage.iters[iid]
                    if real_it.range is None:
                        continue
                    real_ext = int(real_it.range.extent)
                    sym_it.extent = self._recover_iter_extent(
                        sid, iid, real_ext,
                        cr_sym_candidates=cr_sym_candidates,
                        cr_stencil=cr_stencil,
                        cr_ordered_splits=cr_ordered_splits,
                    )

    # ─── AnnotationStep ───
    def _apply_annotation(self, step):
        self.s.stages[step.stage_id].iters[step.iter_id].annotation = int(step.annotation)

    # ─── FuseStep ───
    def _apply_fuse(self, step, step_idx):
        s = self.s
        sid = step.stage_id
        fused_ids = [int(x) for x in step.fused_ids]
        stage = s.stages[sid]

        if not fused_ids:
            new_it = SymIter("", SymExpr(1), annotation=0, iter_kind=3)
            stage.iters.insert(0, new_it)
            return

        self._restore_stage_extents_if_needed(sid, step_idx)

        new_name = "@".join(stage.iters[fid].name for fid in fused_ids) + "@"

        new_extent = SymExpr(1)
        all_defined = True
        new_iter_kind = stage.iters[fused_ids[0]].iter_kind
        for fid in fused_ids:
            it = stage.iters[fid]
            if it.extent is not None:
                new_extent = SymExpr.mul(new_extent, it.extent)
            else:
                all_defined = False
            if it.iter_kind != new_iter_kind:
                new_iter_kind = 2

        new_it = SymIter(new_name,
                         new_extent if all_defined else None,
                         annotation=0, iter_kind=new_iter_kind)

        if ".shared" in stage.op_name and sid not in s._shared_fused_extents:
            shared_extent = self._product_of_defined_iters(stage)
            if shared_extent is not None:
                s._shared_fused_extents[sid] = SymExpr(shared_extent.val)

        begin = fused_ids[0]
        end = fused_ids[-1]
        new_iters = stage.iters[:begin] + [new_it] + stage.iters[end + 1:]
        stage.iters = new_iters

        removed = len(fused_ids) - 1
        for other_stage in s.stages:
            if (other_stage.compute_at == CA_ITER and
                other_stage.attach_stage_id == sid):
                old_aid = other_stage.attach_iter_id
                if old_aid > end:
                    other_stage.attach_iter_id = old_aid - removed
                elif old_aid >= begin:
                    other_stage.attach_iter_id = begin

    # ─── PragmaStep ───
    def _apply_pragma(self, step, step_idx):
        s = self.s
        sid = step.stage_id
        pragma_type = str(step.pragma_type)
        if pragma_type.startswith("auto_unroll_max_step"):
            parts = pragma_type.split("$")
            if len(parts) == 2:
                val = int(parts[1])
                sym_name = f"ur_{step_idx}"
                s.sym_map[sym_name] = val
                s.stages[sid].auto_unroll_max_step = SymExpr(sym_name)
        elif pragma_type == "debug_skip_region":
            s.stages[sid].compute_at = CA_ROOT
            s.stages[sid].attach_stage_id = None
            s.stages[sid].attach_iter_id = None

    # ─── ReorderStep ───
    def _apply_reorder(self, step):
        s = self.s
        sid = step.stage_id
        after_ids = [int(x) for x in step.after_ids]
        stage = s.stages[sid]
        old_iters = stage.iters
        stage.iters = [old_iters[i] for i in after_ids]
        old_to_new = {old_id: new_id for new_id, old_id in enumerate(after_ids)}
        for other_stage in s.stages:
            if (other_stage.compute_at == CA_ITER and
                other_stage.attach_stage_id == sid and
                other_stage.attach_iter_id in old_to_new):
                other_stage.attach_iter_id = old_to_new[other_stage.attach_iter_id]

    # ─── SplitStep ───
    def _apply_split(self, step, step_idx):
        s = self.s
        sid = step.stage_id
        iid = step.iter_id
        lengths = list(step.lengths)
        inner_to_outer = bool(step.inner_to_outer)

        stage = s.stages[sid]
        self._restore_stage_extents_if_needed(sid, step_idx)
        if ".shared" in stage.op_name and sid not in s._shared_fused_extents:
            shared_extent = self._product_of_defined_iters(stage)
            if shared_extent is not None:
                s._shared_fused_extents[sid] = SymExpr(shared_extent.val)

        orig_iter = stage.iters[iid]

        if orig_iter.extent is not None:
            tosplit_extent = orig_iter.extent
        elif step.extent is not None:
            tosplit_extent = SymExpr(int(step.extent))
        else:
            tosplit_extent = None
        if tosplit_extent is not None:
            s._split_step_extents[step_idx] = SymExpr(tosplit_extent.val)

        sym_lengths = []
        for li, length in enumerate(lengths):
            val = int(length) if length is not None else None
            sym_name = f"sp_{step_idx}_{li}"
            s.sym_map[sym_name] = val
            sym_lengths.append(SymExpr(sym_name))

        sym_prod = SymExpr.product(sym_lengths)
        s._split_sym_products[(sid, step_idx)] = sym_prod

        outs = []
        if inner_to_outer:
            for i in range(len(lengths)):
                li = len(lengths) - i - 1
                name = f"{orig_iter.name}.{len(lengths) - i}"
                sym_ext = sym_lengths[li]
                if tosplit_extent is not None:
                    sym_ext = self._clamp_positive_split_extent(sym_ext, tosplit_extent)
                outs.append(SymIter(name, sym_ext, annotation=0, iter_kind=orig_iter.iter_kind))
                if tosplit_extent is not None:
                    tosplit_extent = SymExpr.ceildiv(tosplit_extent, sym_ext)
                else:
                    tosplit_extent = None
            outs.append(SymIter(f"{orig_iter.name}.0", tosplit_extent,
                                annotation=0, iter_kind=orig_iter.iter_kind))
            outs = list(reversed(outs))
        else:
            for i in range(len(lengths)):
                name = f"{orig_iter.name}.{i}"
                sym_ext = sym_lengths[i]
                if tosplit_extent is not None:
                    sym_ext = self._clamp_positive_split_extent(sym_ext, tosplit_extent)
                outs.append(SymIter(name, sym_ext, annotation=0, iter_kind=orig_iter.iter_kind))
                if tosplit_extent is not None:
                    tosplit_extent = SymExpr.ceildiv(tosplit_extent, sym_ext)
                else:
                    tosplit_extent = None
            outs.append(SymIter(f"{orig_iter.name}.{len(lengths)}", tosplit_extent,
                                annotation=0, iter_kind=orig_iter.iter_kind))

        new_iters = stage.iters[:iid] + outs + stage.iters[iid + 1:]
        stage.iters = new_iters

        shift = len(lengths)
        for other_stage in s.stages:
            if (other_stage.compute_at == CA_ITER and
                other_stage.attach_stage_id == sid and
                other_stage.attach_iter_id >= iid):
                other_stage.attach_iter_id += shift

    # ─── FollowSplitStep ───
    def _apply_follow_split(self, step, all_steps, step_idx):
        s = self.s
        sid = step.stage_id
        iid = step.iter_id
        src_step_id = step.src_step_id
        n_split = int(step.n_split)

        src_step = all_steps[src_step_id]
        src_lengths = list(src_step.lengths)

        extracted = []
        for j in range(min(n_split - 1, len(src_lengths))):
            extracted.append(src_lengths[j])
        last = 1
        j_start = n_split - 1
        all_defined = True
        for j in range(j_start, len(src_lengths)):
            if src_lengths[j] is not None:
                last *= int(src_lengths[j])
            else:
                all_defined = False
                break
        extracted.append(last if all_defined else None)

        stage = s.stages[sid]
        self._restore_stage_extents_if_needed(sid, step_idx)

        orig_iter = stage.iters[iid]
        orig_extent = None
        if orig_iter.extent is not None and orig_iter.extent.is_concrete:
            orig_extent = orig_iter.extent.val

        tosplit_extent = SymExpr(orig_extent) if orig_extent is not None else \
                         (orig_iter.extent if orig_iter.extent is not None else None)
        sym_lengths = []
        for li in range(len(extracted)):
            if li < n_split - 1:
                src_sym_name = f"sp_{src_step_id}_{li}"
            else:
                if j_start < len(src_lengths) and j_start == len(src_lengths) - 1:
                    src_sym_name = f"sp_{src_step_id}_{j_start}"
                else:
                    parts = [f"sp_{src_step_id}_{j}" for j in range(j_start, len(src_lengths))]
                    src_sym_name = "*".join(parts) if parts else str(extracted[li])
            sym_lengths.append(SymExpr(src_sym_name))

        outs = []
        for i in range(len(extracted)):
            li = len(extracted) - i - 1
            name = f"{orig_iter.name}.{len(extracted) - i}"
            sym_ext = sym_lengths[li]
            if tosplit_extent is not None:
                sym_ext = self._clamp_positive_split_extent(sym_ext, tosplit_extent)
            outs.append(SymIter(name, sym_ext, annotation=0, iter_kind=orig_iter.iter_kind))
            if tosplit_extent is not None:
                tosplit_extent = SymExpr.ceildiv(tosplit_extent, sym_ext)
            else:
                tosplit_extent = None
        outs.append(SymIter(f"{orig_iter.name}.0", tosplit_extent,
                            annotation=0, iter_kind=orig_iter.iter_kind))
        outs = list(reversed(outs))

        new_iters = stage.iters[:iid] + outs + stage.iters[iid + 1:]
        stage.iters = new_iters

        shift = len(extracted)
        for other_stage in s.stages:
            if (other_stage.compute_at == CA_ITER and
                other_stage.attach_stage_id == sid and
                other_stage.attach_iter_id >= iid):
                other_stage.attach_iter_id += shift

    # ─── FollowFusedSplitStep ───
    def _apply_follow_fused_split(self, step, all_steps, step_idx):
        s = self.s
        sid = step.stage_id
        iid = step.iter_id
        src_step_ids = [int(x) for x in step.src_step_ids]
        level = int(step.level)
        factor_or_nparts = bool(step.factor_or_nparts)

        stage = s.stages[sid]
        self._restore_stage_extents_if_needed(sid, step_idx)

        orig_iter = stage.iters[iid]
        orig_extent = None
        if orig_iter.extent is not None and orig_iter.extent.is_concrete:
            orig_extent = orig_iter.extent.val

        tosplit_extent = SymExpr(orig_extent) if orig_extent is not None else \
                         (orig_iter.extent if orig_iter.extent is not None else None)

        src_sym_parts = []
        for sid_ref in src_step_ids:
            src_sym_parts.append(f"sp_{sid_ref}_{level}")
        fused_sym_expr = SymExpr("*".join(src_sym_parts) if len(src_sym_parts) > 1 else src_sym_parts[0])

        if factor_or_nparts:
            inner_ext = fused_sym_expr
            outer_ext = SymExpr.ceildiv(tosplit_extent, fused_sym_expr) if tosplit_extent else None
            outs = [
                SymIter(f"{orig_iter.name}.0", outer_ext, annotation=0, iter_kind=orig_iter.iter_kind),
                SymIter(f"{orig_iter.name}.1", inner_ext, annotation=0, iter_kind=orig_iter.iter_kind),
            ]
        else:
            outer_ext = fused_sym_expr
            inner_ext = SymExpr.ceildiv(tosplit_extent, fused_sym_expr) if tosplit_extent else None
            outs = [
                SymIter(f"{orig_iter.name}.0", outer_ext, annotation=0, iter_kind=orig_iter.iter_kind),
                SymIter(f"{orig_iter.name}.1", inner_ext, annotation=0, iter_kind=orig_iter.iter_kind),
            ]

        new_iters = stage.iters[:iid] + outs + stage.iters[iid + 1:]
        stage.iters = new_iters

        for other_stage in s.stages:
            if (other_stage.compute_at == CA_ITER and
                other_stage.attach_stage_id == sid and
                other_stage.attach_iter_id >= iid):
                other_stage.attach_iter_id += 1

    # ─── StorageAlignStep ───
    def _apply_storage_align(self, step):
        self.s.stages[step.stage_id].storage_offset = step.offset

    # ─── ComputeAtStep ───
    def _apply_compute_at(self, step):
        s = self.s
        sid = step.stage_id
        target_sid = step.target_stage_id
        target_iid = step.target_iter_id
        stage = s.stages[sid]

        for iid, it in enumerate(stage.iters):
            if it.extent is not None and not it.extent.is_concrete:
                s._ca_saved_extents[(sid, iid)] = SymExpr(it.extent.val)
            it.extent = None
        if ".shared" in stage.op_name:
            s._shared_fused_extents.pop(sid, None)

        stage.compute_at = CA_ITER
        stage.attach_stage_id = target_sid
        stage.attach_iter_id = target_iid

    # ─── ComputeInlineStep ───
    def _apply_compute_inline(self, step):
        stage = self.s.stages[step.stage_id]
        stage.compute_at = CA_INLINED
        stage.attach_stage_id = None
        stage.attach_iter_id = None

    # ─── ComputeRootStep ───
    def _apply_compute_root(self, step):
        s = self.s
        sid = step.stage_id
        stage = s.stages[sid]
        for iid, it in enumerate(stage.iters):
            if it.extent is not None and not it.extent.is_concrete:
                s._ca_saved_extents[(sid, iid)] = SymExpr(it.extent.val)
            it.extent = None
        if ".shared" in stage.op_name:
            s._shared_fused_extents.pop(sid, None)
        stage.compute_at = CA_ROOT
        stage.attach_stage_id = None
        stage.attach_iter_id = None

    # ─── CacheReadStep ───
    def _apply_cache_read(self, step, state, step_idx):
        s = self.s
        sid = step.stage_id
        added_stage_id = sid + 1

        ps_after = tvm._ffi.get_global_func("auto_scheduler.ReplayStepsPartial")(
            s.compute_dag, state, step_idx + 1)

        s.stages[sid].op_name = ps_after.stages[sid].op.name
        s.stages.insert(added_stage_id, self._clone_real_stage(ps_after.stages[added_stage_id]))

        s._shift_ca_saved_extents(added_stage_id)

        reader_ids = [int(x) for x in step.reader_stage_ids]
        if reader_ids:
            consumer_sid = reader_ids[0]
            if consumer_sid >= added_stage_id:
                consumer_sid += 1
            s._cache_read_consumer[added_stage_id] = consumer_sid
            self._analyze_cache_read_stencil(added_stage_id, sid, consumer_sid)

        for other_stage in s.stages:
            if other_stage.compute_at == CA_ITER:
                if other_stage.attach_stage_id is not None and other_stage.attach_stage_id >= added_stage_id:
                    other_stage.attach_stage_id += 1

        for i in range(added_stage_id + 1, len(s.stages)):
            real_stage = ps_after.stages[i]
            s.stages[i].op_name = real_stage.op.name

    def _analyze_cache_read_stencil(self, cr_stage_id, orig_tensor_sid, consumer_sid):
        s = self.s

        cr_name = s.stages[cr_stage_id].op_name
        orig_name = cr_name.rsplit(".", 1)[0] if "." in cr_name else cr_name

        consumer_name = s.stages[consumer_sid].op_name
        consumer_orig_name = consumer_name.rsplit(".", 1)[0] if "." in consumer_name else consumer_name

        consumer_op = None
        for op in s.compute_dag.ops:
            if op.name == consumer_orig_name or op.name == consumer_name:
                if hasattr(op, 'body'):
                    consumer_op = op
                    break

        if consumer_op is None or not hasattr(consumer_op, 'body') or len(consumer_op.body) == 0:
            return

        producer_loads = self._find_producer_loads(consumer_op.body[0], orig_name)
        if not producer_loads:
            return

        pl = producer_loads[0]

        spatial_vars = {str(ax.var.name): i for i, ax in enumerate(consumer_op.axis)}
        reduce_vars = {str(ax.var.name): len(consumer_op.axis) + i
                      for i, ax in enumerate(consumer_op.reduce_axis)}

        stencil_info = {}
        for ax_idx, idx_expr in enumerate(pl.indices):
            result = self._analyze_index_expr(idx_expr, consumer_op.axis, consumer_op.reduce_axis)
            if result is None:
                continue
            pattern_kind, pattern_value, sp_name, rd_name = result
            sp_order = spatial_vars.get(sp_name) if sp_name else None
            rd_order = reduce_vars.get(rd_name) if rd_name else None
            stencil_info[ax_idx] = (pattern_kind, pattern_value, sp_order, rd_order)

        if stencil_info:
            s._cache_read_stencil_info[cr_stage_id] = stencil_info

    @staticmethod
    def _find_producer_loads(expr, tensor_name):
        results = []
        def visit(e):
            if isinstance(e, tir.ProducerLoad):
                if str(e.producer.name) == tensor_name:
                    results.append(e)
                return
            if isinstance(e, tir.Reduce):
                for src in e.source:
                    visit(src)
                return
            for attr in ['a', 'b']:
                if hasattr(e, attr):
                    visit(getattr(e, attr))
            if hasattr(e, 'args') and not isinstance(e, (tir.Var, tir.IntImm)):
                for arg in e.args:
                    visit(arg)
        visit(expr)
        return results

    @staticmethod
    def _analyze_index_expr(expr, spatial_axes, reduce_axes):
        spatial_names = {str(v.var.name) for v in spatial_axes}
        reduce_names = {str(v.var.name) for v in reduce_axes}

        if isinstance(expr, tir.Var):
            name = str(expr.name)
            if name in spatial_names:
                return ("direct", 0, name, None)
            elif name in reduce_names:
                return ("direct", 0, None, name)
            return None

        if isinstance(expr, tir.Add):
            a, b = expr.a, expr.b
            def extract_mul_var(e):
                if isinstance(e, tir.Mul):
                    if isinstance(e.a, tir.Var) and isinstance(e.b, tir.IntImm):
                        return (str(e.a.name), int(e.b))
                    if isinstance(e.b, tir.Var) and isinstance(e.a, tir.IntImm):
                        return (str(e.b.name), int(e.a))
                elif isinstance(e, tir.Var):
                    return (str(e.name), 1)
                return None

            def extract_grouped_var(e):
                if not isinstance(e, tir.Mul):
                    return None
                cases = ((e.a, e.b), (e.b, e.a))
                for lhs, rhs in cases:
                    if not isinstance(lhs, tir.FloorDiv):
                        continue
                    if not isinstance(rhs, tir.IntImm):
                        continue
                    if not isinstance(lhs.a, tir.Var) or not isinstance(lhs.b, tir.IntImm):
                        continue
                    block = int(rhs)
                    if block != int(lhs.b):
                        continue
                    return (str(lhs.a.name), block)
                return None

            mul_info = extract_mul_var(a)
            if mul_info and isinstance(b, tir.Var):
                sp_name, stride = mul_info
                rd_name = str(b.name)
                if sp_name in spatial_names and rd_name in reduce_names:
                    return ("linear", stride, sp_name, rd_name)

            mul_info = extract_mul_var(b)
            if mul_info and isinstance(a, tir.Var):
                sp_name, stride = mul_info
                rd_name = str(a.name)
                if sp_name in spatial_names and rd_name in reduce_names:
                    return ("linear", stride, sp_name, rd_name)

            grouped_info = extract_grouped_var(a)
            if grouped_info and isinstance(b, tir.Var):
                sp_name, block = grouped_info
                rd_name = str(b.name)
                if sp_name in spatial_names and rd_name in reduce_names:
                    return ("grouped", block, sp_name, rd_name)

            grouped_info = extract_grouped_var(b)
            if grouped_info and isinstance(a, tir.Var):
                sp_name, block = grouped_info
                rd_name = str(a.name)
                if sp_name in spatial_names and rd_name in reduce_names:
                    return ("grouped", block, sp_name, rd_name)

            if isinstance(a, tir.Var) and isinstance(b, tir.Var):
                a_name, b_name = str(a.name), str(b.name)
                if a_name in spatial_names and b_name in reduce_names:
                    return ("linear", 1, a_name, b_name)
                if b_name in spatial_names and a_name in reduce_names:
                    return ("linear", 1, b_name, a_name)

        return None

    # ─── CacheWriteStep ───
    def _apply_cache_write(self, step, state, step_idx):
        s = self.s
        sid = step.stage_id
        ps_after = tvm._ffi.get_global_func("auto_scheduler.ReplayStepsPartial")(
            s.compute_dag, state, step_idx + 1)
        added_ops = len(ps_after.stages) - len(s.stages)
        if added_ops < 1:
            raise RuntimeError(f"Unexpected CacheWrite added_ops={added_ops} at stage {sid}")
        if added_ops > 2:
            raise RuntimeError(f"Unsupported CacheWrite added_ops={added_ops} at stage {sid}")

        s.stages.insert(sid, self._clone_real_stage(ps_after.stages[sid]))
        s.stages[sid + 1] = self._clone_real_stage(ps_after.stages[sid + 1])

        next_stage_id = sid + 2
        if added_ops == 2:
            s.stages.insert(next_stage_id, self._clone_real_stage(ps_after.stages[next_stage_id]))
            next_stage_id += 1

        s._shift_ca_saved_extents(sid, offset=added_ops)

        for other_stage in s.stages:
            if other_stage.compute_at == CA_ITER:
                if other_stage.attach_stage_id is not None and other_stage.attach_stage_id >= sid:
                    other_stage.attach_stage_id += added_ops

        for i in range(next_stage_id, len(s.stages)):
            if i < len(ps_after.stages):
                real_stage = ps_after.stages[i]
                s.stages[i].op_name = real_stage.op.name
