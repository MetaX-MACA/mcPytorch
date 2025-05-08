import gc
import unittest

import torch
import torch.cuda
from torch._six import inf, nan

from torch.testing._internal.common_utils import TestCase, run_tests
import torch.backends

"""
Test cases extracted from pytorch/test/test_cuda.py
"""
class TestGraph(TestCase):
    FIFTY_MIL_CYCLES = 500
    def setUp(self):
        super(TestGraph, self).setUp()

    #@unittest.skipIf(True, "TODO(liuyuxin): not available!")
    def test_graph_capture_simple(self):
        s = torch.cuda.Stream()

        with torch.cuda.stream(s):
            a = torch.full((1000,), 1, device="cuda")
            g = torch.cuda.CUDAGraph()
            torch.cuda.empty_cache()
            g.capture_begin()
            b = a
            for _ in range(10):
                b = b + 1
            g.capture_end()
        torch.cuda.current_stream().wait_stream(s)

        g.replay()

        self.assertTrue(b.sum().item() == 11000.)

    #@unittest.skipIf(True, "TODO(liuyuxin): not available!")
    def test_graph_two_successive(self):
        torch.cuda.empty_cache()

        size = 1000
        kSmallBuffer = 2097152

        def func_with_temps(t, val):
            x = t.clone() + val
            y = t.clone() + val
            return x + y

        s = torch.cuda.Stream()

        for share_mem in ("Don't share", "via pool()", "via graph_pool_handle()"):
            g0 = torch.cuda.CUDAGraph()
            g1 = torch.cuda.CUDAGraph()

            a = torch.ones((size,), device="cuda")

            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                g0_args = (torch.cuda.graph_pool_handle(),) if share_mem == "via graph_pool_handle()" else ()
                g0.capture_begin(*g0_args)
                b = a.clone()
                for _ in range(5):
                    b = func_with_temps(b, 1)
                g0.capture_end()

                g1_args = (g0.pool(),) if share_mem == "via pool()" else g0_args
                g1.capture_begin(*g1_args)
                for _ in range(5):
                    b = func_with_temps(b, 1)
                g1.capture_end()
            torch.cuda.current_stream().wait_stream(s)

            # mixes unrelated eager ops with replays
            c = a.clone()
            for _ in range(2):
                c = func_with_temps(c, 3)
            g0.replay()
            for _ in range(2):
                c = func_with_temps(c, 3)
            g1.replay()
            for _ in range(2):
                c = func_with_temps(c, 3)

            self.assertEqual(b.sum().item(), size * 3070)
            self.assertEqual(c.sum().item(), size * 442)

            if share_mem != "Don't share":
                print(share_mem)
                self.assertEqual(reserved_no_sharing - torch.cuda.memory_stats()["reserved_bytes.all.current"],
                                 kSmallBuffer)
            else:
                reserved_no_sharing = torch.cuda.memory_stats()["reserved_bytes.all.current"]

            del a, b, c, g0, g1
            # Tensors used across streams (a and b) were held until just now, so no need to call record_stream on them.
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    #@unittest.skipIf(True, "TODO(liuyuxin): not available!")
    def test_graph_three_successive(self):
        torch.cuda.empty_cache()

        size = 1000

        s = torch.cuda.Stream()

        for share_mem in ("Don't share", "via pool()", "via graph_pool_handle()"):
            a = torch.ones((size,), device="cuda")

            g0 = torch.cuda.CUDAGraph()
            g1 = torch.cuda.CUDAGraph()
            g2 = torch.cuda.CUDAGraph()

            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                g0_args = (torch.cuda.graph_pool_handle(),) if share_mem == "via graph_pool_handle()" else ()
                g0.capture_begin(*g0_args)
                b = a.clone()
                c = b + 1
                d = b + 2
                g0.capture_end()

                args = (g0.pool(),) if share_mem == "via pool()" else g0_args

                g1.capture_begin(*args)
                e = c + 3
                del c
                g1.capture_end()

                g2.capture_begin(*args)
                f = d + 4
                g2.capture_end()
            torch.cuda.current_stream().wait_stream(s)

            # Tests that replaying in capture order is valid
            g0.replay()
            g1.replay()
            g2.replay()

            self.assertEqual(e.sum().item(), size * 5)
            self.assertEqual(f.sum().item(), size * 7)

            # Tests that replaying as g0, g2, g1 is only valid if they don't share a pool
            g0.replay()
            g2.replay()
            g1.replay()

            # If share_mem is True, g2's capture should have reused c's memory for f. We replayed g2 then g1,
            # so we expect g1's captured "e = c + 3" mistakenly filled e with "f's vals + 3".
            self.assertEqual(e.sum().item(), size * (7 + 3) if share_mem != "Don't share" else size * 5)
            self.assertEqual(f.sum().item(), size * 7)

            del a, b, d, e, f, g0, g1, g2
            # Tensors used across streams (a, e, f) were held until just now, so no need to call record_stream on them.
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    #@unittest.skipIf(True, "TODO(liuyuxin): not available!")
    def test_graph_concurrent_replay(self):
        torch.cuda.empty_cache()

        size = 1000000  # largeish to help expose race conditions

        def func_with_temps(t, val):
            x = t.clone() + val
            y = t.clone() + val
            return x + y

        s = torch.cuda.Stream()

        for share_mem in ("Don't share", "via pool()", "via graph_pool_handle()"):
            g0 = torch.cuda.CUDAGraph()
            g1 = torch.cuda.CUDAGraph()

            s0 = torch.cuda.Stream()
            s1 = torch.cuda.Stream()

            a = torch.ones((size,), device="cuda")

            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                g0_args = (torch.cuda.graph_pool_handle(),) if share_mem == "via graph_pool_handle()" else ()
                g0.capture_begin(*g0_args)
                b = a.clone()
                for _ in range(5):
                    b = func_with_temps(b, 1)
                g0.capture_end()

                g1_args = (g0.pool(),) if share_mem == "via pool()" else g0_args
                g1.capture_begin(*g1_args)
                c = a.clone()
                for _ in range(5):
                    c = func_with_temps(c, 2)
                g1.capture_end()

            # To reproduce data corruption, I need g0 and g1's kernels to run concurrently.
            # But replay() (especially cudaGraphLaunch) can incur significant CPU overhead.
            # The following pattern helps align device-side execution of g0 and g1's kernels.
            torch.cuda.synchronize()
            with torch.cuda.stream(s0):
                torch.cuda._sleep(100)
                s1.wait_stream(s0)
                g0.replay()
            with torch.cuda.stream(s1):
                g1.replay()
            torch.cuda.current_stream().wait_stream(s0)
            torch.cuda.current_stream().wait_stream(s1)

            if share_mem != "Don't share":
                # Confirms concurrent replays using the same mempool corrupted each other.
                self.assertNotEqual(b.sum().item(), size * 94)
                self.assertNotEqual(c.sum().item(), size * 156)
            else:
                # Confirms concurrent replays using different mempools did not corrupt each other.
                self.assertEqual(b.sum().item(), size * 94)
                self.assertEqual(c.sum().item(), size * 156)

            del a, b, c, g0, g1
            # Tensors used across streams (a, b, c) were held until just now, so no need to call record_stream on them.
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    #@unittest.skipIf(True, "TODO(liuyuxin): not available!")
    def test_graph_memory_stats_and_use_result_after_destroy_graph(self):
        kSmallSize = 1048576
        kSmallBuffer = 2097152
        kLargeBuffer = 20971520
        kMinLargeAlloc = 10485760
        kRoundLarge = 2097152

        elem = 4

        # this was annoying to write but stresses the expectations pretty rigorously
        cases = ((512 // elem, 1, kSmallBuffer, kSmallBuffer, "small_pool"),
                 (kSmallSize // elem, 2, 2 * kSmallBuffer, kSmallBuffer, "small_pool"),
                 ((kSmallSize + 512) // elem, 1, kLargeBuffer, kLargeBuffer, "large_pool"),
                 ((kMinLargeAlloc - 512) // elem, 2, 2 * kLargeBuffer, kLargeBuffer, "large_pool"),
                 ((kMinLargeAlloc + 512) // elem, 3,
                  3 * (kRoundLarge * ((kMinLargeAlloc + 512 + kRoundLarge - 1) // kRoundLarge)),
                  kRoundLarge * ((kMinLargeAlloc + 512 + kRoundLarge - 1) // kRoundLarge),
                  "large_pool"),)

        stats_to_check = ("segment.",
                          "reserved_bytes.",
                          "active.",
                          "active_bytes.")

        gc.collect()
        torch.cuda.empty_cache()

        s = torch.cuda.Stream()

        for (numel,
             delta_cudaMallocs,
             delta_cudaMalloc_bytes,
             delta_cudaMalloc_bytes_post_del_g,
             pool_string) in cases:
            if pool_string == "small_pool":
                delta_active_blocks = 2  # one from "b" plus a sneaky one from CUDAGraph's one-element rng offset holder
                delta_active_bytes = numel * elem + 512  # + 512 for CUDAGraph's rng offset holder
            else:
                delta_active_blocks = 1  # We only check the large pool, which isn't affected by rng offset holder
                delta_active_bytes = numel * elem

            g = torch.cuda.CUDAGraph()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                # Allocation stat estimates assume input is created on the same stream as capture_begin()
                # (in other words, the same stream silo as the rng offset holder, which is not allocated from the
                # capture's private pool).
                a = torch.ones((numel,), device="cuda")

                precapture_stats = torch.cuda.memory_stats()

                g.capture_begin()
                b = a.clone()
                for _ in range(5):
                    b = b.clone() + 1
                g.capture_end()
            torch.cuda.current_stream().wait_stream(s)

            gc.collect()

            postcapture_stats = torch.cuda.memory_stats()

            expecteds = (delta_cudaMallocs,
                         delta_cudaMalloc_bytes,
                         delta_active_blocks,
                         delta_active_bytes)
            # Double checks replay and stats before and after a call to empty_cache
            for i in range(2):
                for stat, expected in zip(stats_to_check, expecteds):
                    stat = stat + pool_string + ".current"
                    current = postcapture_stats[stat] - precapture_stats[stat]
                    self.assertEqual(current, expected, "Pre to post capture delta of " +
                                     stat + " = {}, expected = {}, numel = {}".format(current, expected, numel))

                g.replay()
                self.assertEqual(b.sum().item(), 6 * numel)
                if i == 0:
                    torch.cuda.empty_cache()

            del g
            gc.collect()
            torch.cuda.empty_cache()
            postdel_stats = torch.cuda.memory_stats()

            # Uses graph result b after graph has been deleted
            self.assertEqual(b.sum().item(), 6 * numel)

            # b should be the only live reference remaining from the graph's private pool
            expecteds = (1, delta_cudaMalloc_bytes_post_del_g, 1, numel * elem)
            for stat, expected in zip(stats_to_check, expecteds):
                stat = stat + pool_string + ".current"
                current = postdel_stats[stat] - precapture_stats[stat]
                self.assertEqual(current, expected, "Pre capture to post graph delete delta of " +
                                 stat + " = {}, expected = {}, numel = {}".format(current, expected, numel))

            # del a, b before the next case is essential, otherwise overwriting a and b in the next case
            # can throw off its allocation/deallocation counts.
            del a, b
            # Tensors used across streams (a and b) were held until just now, so no need to call record_stream on them.
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    #@unittest.skipIf(True, "TODO(liuyuxin): not available!")
    def test_graph_record_stream(self):
        # Makes sure graph capture defers attempting to reclaim allocations used across streams. See
        # "Q. Why skip process_events if a capture might be underway?" in c10/cuda/CUDACachingAllocator.cpp
        torch.cuda.empty_cache()

        potential_problem = torch.zeros((3,), device="cuda")
        a = torch.zeros((3,), device="cuda")
        s0 = torch.cuda.Stream()
        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()
        g = torch.cuda.CUDAGraph()

        torch.cuda.synchronize()
        with torch.cuda.stream(s0):
            potential_problem.record_stream(s0)
            torch.cuda._sleep(TestGraph.FIFTY_MIL_CYCLES)
            potential_problem.fill_(1.)
        del potential_problem

        with torch.cuda.stream(s1):
            g.capture_begin()
            # potential_problem's allocation should still be outstanding. if DeviceCachingAllocator::malloc
            # mistakenly calls process_events, it will trigger cudaEventQueries on potential_problem's end-of-life
            # event, which will cause the capture to error.
            b = a.clone()

            # Let's also see what happens if we record_stream on a tensor during capture.
            s2.wait_stream(s1)
            with torch.cuda.stream(s2):
                b.fill_(1.)
                b.record_stream(s2)  # dummy record_stream
                del b
            s1.wait_stream(s2)
            g.capture_end()
        torch.cuda.synchronize()

        # dummy allocation triggers process_events, Hopefully successfully processes b's end-of-life event.
        c = torch.zeros((3,), device="cuda")

    @unittest.skipIf(True, "TODO(liuyuxin): not available!")
    def test_graph_cudnn_dropout(self):
        # Tests the interaction of cuda graph capture with DropoutState's syncs in ATen/native/cudnn/RNN.cpp.
        # In particular, if user runs a sequence of captured and noncaptured cudnn rnns, DropoutState should
        # avoid syncing noncapturing streams with captured events or vice versa.
        torch.cuda.empty_cache()

        model = torch.nn.LSTM(4, 4, 2, dropout=0.5).cuda()
        x = torch.ones(5, 12, 4, device="cuda")

        y = model(x)

        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            g.capture_begin()
            y = model(x)
            g.capture_end()
        torch.cuda.current_stream().wait_stream(s)

        y = model(x)

    #@unittest.skipIf(True, "TODO(liuyuxin): not available!")
    def test_graph_rng_functional(self):
        ops_with_kwargs = ((torch.nn.functional.dropout, {"p": 0.1}),
                           (torch.nn.functional.rrelu, {"training": True}),)
        size = 10000

        def run(op, kwargs):
            a = torch.randn((size,), device="cuda", dtype=torch.float)

            # Control
            torch.cuda.manual_seed(5)
            eager_out = a
            for _ in range(6):
                eager_out = op(eager_out, **kwargs)

            graph_in = a.clone()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                torch.cuda.manual_seed(5)

                g = torch.cuda.CUDAGraph()
                torch.cuda.empty_cache()
                g.capture_begin()
                graph_out = graph_in
                for _ in range(2):
                    graph_out = op(graph_out, **kwargs)
                g.capture_end()
            torch.cuda.current_stream().wait_stream(stream)

            # Runs a graphed->eager->graphed sequence of RNG ops.
            # replay() plays 2 invocations of the op, so the sequence has 6
            # invocations total, matching Control.
            # replay() reads from graph_in and writes to graph_out.
            g.replay()
            out = op(graph_out, **kwargs)
            out = op(out, **kwargs)
            graph_in.copy_(out)
            g.replay()

            # If replay() updated RNG state correctly, graph_out
            # should now hold data equal to eager_out.
            try:
                self.assertEqual(eager_out, graph_out)
            except Exception as e:
                raise RuntimeError("Failed on ", op) from e

            # We hold references to all tensors used across streams up til this sync,
            # so no need to call record_stream on those tensors.
            torch.cuda.synchronize()

        for op, kwargs in ops_with_kwargs:
            run(op, kwargs)

    #@unittest.skipIf(True, "TODO(liuyuxin): not available!")
    def test_graph_rng_distributions(self):
        size = 10000
        input = torch.rand((size,), device="cuda", dtype=torch.float)
        alloc = torch.empty((size,), device="cuda", dtype=torch.float)

        # Torch ops to test with sample args (tuple) and kwargs (dict)
        torch_with_args = (("bernoulli", (input.clone(),), {}),
                           # multinomial uses some uncapturable CUDA calls.
                           # TODO: reenable multinomial tests if/when the implementation is capturable.
                           # ("multinomial", (input.clone(), size, True), {}),
                           # ("multinomial", (input.clone(), size // 2, False), {}),
                           # TODO: reenable normal test, where std is a device
                           # tensor, when graph test failures are fixed
                           # ("normal", (input.clone() + 1, input.clone()), {}),
                           ("normal", (input.clone() + 1, 1.0), {}),
                           ("poisson", (input.clone(),), {}),
                           ("rand", (size,), {"device": "cuda", "dtype": torch.float}),
                           ("randint", (0, 3, (size,)), {"device": "cuda", "dtype": torch.float}),
                           ("randn", (size,), {"device": "cuda", "dtype": torch.float}),)

        # Tensor methods to test with sample args (tuple)
        tensor_with_args = (("bernoulli_", (input.clone(),)),
                            ("cauchy_", ()),
                            ("exponential_", ()),
                            ("geometric_", (0.3,)),
                            ("log_normal_", ()),
                            ("normal_", ()),
                            ("random_", ()),
                            ("uniform_", ()),)

        def run(module, op, args, kwargs):
            torch.cuda.manual_seed(5)

            # Each path runs a dummy op to increment the state a bit before creating controls.
            if (module == "torch"):
                dummy = getattr(torch, op)(*args, **kwargs)
                control1 = getattr(torch, op)(*args, **kwargs)
                control2 = getattr(torch, op)(*args, **kwargs)
            else:
                dummy = alloc.clone()
                control1 = alloc.clone()
                control2 = alloc.clone()
                getattr(dummy, op)(*args)
                getattr(control1, op)(*args)
                getattr(control2, op)(*args)

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                torch.cuda.manual_seed(5)

                g = torch.cuda.CUDAGraph()
                torch.cuda.empty_cache()
                if (module == "torch"):
                    g.capture_begin()
                    t1 = getattr(torch, op)(*args, **kwargs)
                    t2 = getattr(torch, op)(*args, **kwargs)
                    g.capture_end()
                else:
                    t1 = alloc.clone()
                    t2 = alloc.clone()
                    g.capture_begin()
                    getattr(t1, op)(*args)
                    getattr(t2, op)(*args)
                    g.capture_end()
            torch.cuda.current_stream().wait_stream(stream)

            try:
                self.assertNotEqual(control1, t1)
                self.assertNotEqual(control2, t2)
            except Exception as e:
                raise RuntimeError("Failed on " + module + "." + op) from e

            # Runs a dummy op prelude, as for controls, to make sure replay()
            # picks up the dummy op's state increment.
            if module == "torch":
                dummy = getattr(torch, op)(*args, **kwargs)
            else:
                dummy = alloc.clone()
                getattr(dummy, op)(*args)

            # Runs RNG ops that fill t1 and t2.
            g.replay()

            try:
                self.assertEqual(control1, t1)
                self.assertEqual(control2, t2)
            except Exception as e:
                raise RuntimeError("Failed on " + module + "." + op) from e

            # We hold references to all tensors used across streams up til this sync,
            # so no need to call record_stream on those tensors.
            torch.cuda.synchronize()

        for op_with_args in torch_with_args:
            run("torch", *op_with_args)

        for meth_with_args in tensor_with_args:
            # Adds an empty dict for kwargs, which none of the Tensor methods use
            run("Tensor", *(meth_with_args + ({},)))

    @unittest.skipIf(True, "TODO(liuyuxin): not available!")
    def test_graph_grad_scaling(self):
        torch.cuda.empty_cache()

        scaler = torch.cuda.amp.GradScaler(init_scale=4.)
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()

        weight = torch.ones((100,), device="cuda", requires_grad=True)
        opt = torch.optim.SGD([weight], lr=0.1)
        static_input = torch.ones_like(weight)
        static_grad = torch.ones_like(weight)

        # warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            loss = (weight.half() * static_input).sum()
            scaler.scale(loss).backward()
        torch.cuda.current_stream().wait_stream(s)

        opt.zero_grad(set_to_none=True)

        # capture
        with torch.cuda.graph(g):
            loss = (weight.half() * static_input).sum()
            scaler.scale(loss).backward()

        input_vals = [5, 20000, 5, 40000]
        # If the scale gets updated properly, these are the scale, growth tracker,
        # and grad values we expect.
        expected_scales = [4, 2, 2, 1]
        expected_growth_trackers = [1, 0, 1, 0]
        expected_grad_vals = [5 * 4, float("inf"), 5 * 2, float("inf")]

        for data, scale, growth_tracker, grad_val in zip(input_vals,
                                                         expected_scales,
                                                         expected_growth_trackers,
                                                         expected_grad_vals):
            static_input.fill_(data)
            g.replay()
            self.assertEqual(weight.grad, torch.full_like(weight.grad, grad_val))
            scaler.step(opt)
            scaler.update()
            self.assertEqual(scaler._scale, scale)
            self.assertEqual(scaler._growth_tracker, growth_tracker)

    @unittest.skipIf(True, "TODO(liuyuxin): not available!")
    def test_graph_make_graphed_callables(self):
        torch.manual_seed(5)
        torch.cuda.manual_seed(5)

        N, D_in, H, D_out = 1, 16, 8, 4

        models = []
        for _ in range(2):
            model_section1 = torch.nn.Sequential(torch.nn.Linear(D_in, H),
                                                 torch.nn.Dropout(p=0.1)).cuda()
            model_section2 = torch.nn.Sequential(torch.nn.Linear(H, D_out),
                                                 torch.nn.Dropout(p=0.2)).cuda()
            models.append(torch.nn.Sequential(model_section1, model_section2))

        model_graphed = models[0]
        model_control = models[1]

        model_graphed.load_state_dict(model_control.state_dict())

        opt_graphed = torch.optim.SGD(model_graphed.parameters(), lr=0.1)
        opt_control = torch.optim.SGD(model_control.parameters(), lr=0.1)

        x = torch.randn(N, D_in, device='cuda')
        h = torch.randn(N, H, device='cuda', requires_grad=True)
        y_pred = torch.randn(N, D_out, device='cuda', requires_grad=True)
        y = torch.randn(N, D_out, device='cuda')

        loss_fn_control = torch.nn.functional.mse_loss
        relu_control = torch.nn.functional.relu

        # This is a good stress test. It graphs four callables: two Modules and two python functions.
        model_graphed[0], model_graphed[1], relu_graphed, loss_fn_graphed = \
            torch.cuda.make_graphed_callables((model_graphed[0], model_graphed[1], relu_control, loss_fn_control),
                                              ((x,), (h,), (y_pred,), (y_pred, y)))

        real_inputs = [torch.rand_like(x) for _ in range(10)]
        real_targets = [torch.rand_like(y) for _ in range(10)]

        for m, opt, relu, loss_fn in zip((model_graphed, model_control),
                                         (opt_graphed, opt_control),
                                         (relu_graphed, relu_control),
                                         (loss_fn_graphed, loss_fn_control)):
            # Resets RNC states before iterations for graphed and ungraphed models,
            # so dropout math should be bitwise identical for both.
            torch.manual_seed(5)
            torch.cuda.manual_seed(5)
            for data, target in zip(real_inputs, real_targets):
                opt.zero_grad(set_to_none=True)
                y_pred = m(data)
                y_pred = relu(y_pred)
                loss = loss_fn(y_pred, target)
                loss.backward()
                opt.step()

        for p, pc in zip(model_graphed.parameters(), model_control.parameters()):
            self.assertEqual(p, pc)

        # We graphed the models in training mode. Eval should still run ungraphed.
        model_graphed.eval()
        model_control.eval()
        self.assertEqual(model_graphed(real_inputs[0]), model_control(real_inputs[0]))

if __name__ == '__main__':
    run_tests()
