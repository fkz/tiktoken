const std = @import("std");
const BlockCount = 12;
const LayerSize = 768;
const Hidden = 4 * LayerSize;
const Vector = [LayerSize]f32;
const Matrix = [LayerSize][LayerSize]u16;
const TokenCount = 50257;
const Heads = 12;

const ThreadCount = @import("build_options").thread_count;
const Prefetch = @import("build_options").prefetch;
const HugePages = @import("build_options").huge_pages;

// Keep both benchmark modes' allocation layouts identical. Whole, aligned
// 2 MiB regions allow even the smaller matrices to use Linux x86 THP and
// keep madvise away from allocator metadata and neighboring allocations.
// The caller's arena owns the backing allocation, including its padding.
fn createOptimizedWeight(comptime T: type, alloc: std.mem.Allocator) !*T {
    if (@import("builtin").os.tag != .linux or !HugePages) return alloc.create(T);
    const huge_page_size = 2 * 1024 * 1024;
    const size = std.mem.alignForward(usize, @sizeOf(T), huge_page_size);
    const bytes = try alloc.alignedAlloc(u8, .fromByteUnits(huge_page_size), size);
    errdefer alloc.free(bytes);
    try std.posix.madvise(bytes.ptr, bytes.len, std.posix.MADV.HUGEPAGE);
    // Fault the padding in too, so the entire final huge page is backed.
    @memset(bytes, 0);
    return @ptrCast(bytes.ptr);
}

test "optimized weight allocation spans huge page boundaries" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const T = [2 * 1024 * 1024 + 64]u8;
    const weight = try createOptimizedWeight(T, arena.allocator());
    if (@import("builtin").os.tag == .linux) {
        try std.testing.expectEqual(@as(usize, 0), @intFromPtr(weight) % (2 * 1024 * 1024));
        try std.testing.expectEqual(@as(u8, 0), weight[weight.len - 1]);
    }
    weight[0] = 17;
    weight[weight.len - 1] = 29;
    try std.testing.expectEqual(@as(u8, 17), weight[0]);
    try std.testing.expectEqual(@as(u8, 29), weight[weight.len - 1]);
}

const BlockData = struct {
    attnNormBias: *align(64) const Vector,
    attnNormWeight: *align(64) const Vector,
    attnQkvBias: *align(64) const [3]Vector,
    attnQkvWeight: *align(64) const [3][LayerSize][LayerSize]u16,
    attnOutputBias: *align(64) const Vector,
    attnOutputWeight: *align(64) const Matrix,
    ffnNormBias: *align(64) const Vector,
    ffnNormWeight: *align(64) const Vector,
    ffnUpBias: *align(64) const [Hidden]f32,
    ffnUpWeight: *align(64) const [Hidden][LayerSize]u16,
    ffnDownBias: *align(64) const Vector,
    ffnDownWeight: *align(64) const [LayerSize][Hidden]u16,
};

// 250MB

gguf: []align(std.heap.page_size_min) u8,

tensors: []align(32) u8,

blocks: [BlockCount]BlockData,
outputNormBias: *align(64) const Vector,
outputNormWeight: *align(64) const Vector,
positionEmbedWeight: *align(64) const [1024]Vector,
tokenEmdebWeight: *align(64) const [TokenCount][LayerSize]u16,

tokens: *const [TokenCount][]const u8,

fn tensorAt(this: *@This(), T: type, offset: u64) *align(64) const T {
    return @ptrCast(@alignCast(this.tensors[offset..][0..@sizeOf(T)]));
}

fn insert(this: *@This(), str: []const u8, offset: u64) !void {
    if (std.mem.eql(u8, str, "output_norm.bias")) {
        this.outputNormBias = this.tensorAt(Vector, offset);
    } else if (std.mem.eql(u8, str, "output_norm.weight")) {
        this.outputNormWeight = this.tensorAt(Vector, offset);
    } else if (std.mem.eql(u8, str, "position_embd.weight")) {
        this.positionEmbedWeight = this.tensorAt([1024]Vector, offset);
    } else if (std.mem.eql(u8, str, "token_embd.weight")) {
        this.tokenEmdebWeight = this.tensorAt([TokenCount][LayerSize]u16, offset);
    } else if (std.mem.eql(u8, str[0..4], "blk.")) {
        const afterIndex: usize = if (str[5] == '.') 5 else if (str[6] == '.') 6 else return error.UnknownTensorName;
        const num = try std.fmt.parseUnsigned(u8, str[4..afterIndex], 10);
        inline for (@typeInfo(BlockData).@"struct".fields) |field| {
            comptime var indexes: [2]usize = undefined;
            comptime var set = 0;
            inline for (field.name, 0..) |c, idx| {
                if (c >= 'A' and c <= 'Z') {
                    indexes[set] = idx;
                    set += 1;
                    if (set == 2) break;
                }
            }
            const n = std.fmt.comptimePrint("{s}_{c}{s}.{c}{s}", .{ field.name[0..indexes[0]], field.name[indexes[0]] + 32, field.name[indexes[0] + 1 .. indexes[1]], field.name[indexes[1]] + 32, field.name[indexes[1] + 1 ..] });
            const T = @typeInfo(field.type).pointer.child;
            if (std.mem.eql(u8, str[afterIndex + 1 ..], n)) {
                @field(this.blocks[num], field.name) = this.tensorAt(T, offset);
                return;
            }
        }
        std.debug.print("{s}", .{str[afterIndex + 1 ..]});
        return error.UnknownTensorName;
    } else {
        return error.UnknownTensorName;
    }
}

fn readValue(reader: *std.Io.Reader, valueType: u32) !void {
    switch (valueType) {
        0 => reader.toss(1),
        1 => reader.toss(1),
        2 => reader.toss(2),
        3 => reader.toss(2),
        4 => reader.toss(4),
        5 => reader.toss(4),
        6 => reader.toss(4),
        7 => reader.toss(1),
        8 => {
            const len = try reader.takeInt(u64, .little);
            reader.toss(len);
        },
        9 => {
            const value = try reader.takeInt(u32, .little);
            const len = try reader.takeInt(u64, .little);
            for (0..len) |_| {
                try readValue(reader, value);
            }
        },
        10 => reader.toss(8),
        11 => reader.toss(8),
        12 => reader.toss(8),
        else => {
            @branchHint(.unlikely);
            return error.UnknownValueType;
        },
    }
}

const Ftype = enum(u32) {
    tF32 = 0,
    tbf16 = 30,
};

fn init(io: std.Io, path: []const u8, alloc: std.mem.Allocator) !@This() {
    const f = try std.Io.Dir.cwd().openFile(io, path, .{});
    errdefer f.close(io);
    const s = try f.stat(io);
    const data = try std.posix.mmap(null, s.size, .{ .READ = true }, .{ .TYPE = .PRIVATE }, f.handle, 0);

    var alignment: u32 = 32;

    var reader = std.Io.Reader.fixed(data);
    if (!std.mem.eql(u8, try reader.take(8), "GGUF\x03\x00\x00\x00")) {
        return error.WrongFile;
    }

    const tensorCount = try reader.takeInt(u64, .little);
    const metadataCount = try reader.takeInt(u64, .little);
    const tokens: *[TokenCount][]const u8 = try alloc.create([TokenCount][]const u8);

    for (0..metadataCount) |_| {
        const len = try reader.takeInt(u64, .little);
        const key = try reader.take(len);
        const valueType = try reader.takeInt(u32, .little);
        if (std.mem.eql(u8, key, "general.alignment")) {
            alignment = try reader.takeInt(u32, .little);
        } else if (std.mem.eql(u8, key, "tokenizer.ggml.tokens")) {
            if (valueType != 9) return error.WrongMetadata;
            const innerValueType = try reader.takeInt(u32, .little);
            const l = try reader.takeInt(u64, .little);
            if (innerValueType != 8 or l != TokenCount) {
                return error.WrongMetadata;
            }
            for (0..TokenCount) |j| {
                const lnd = try reader.takeInt(u64, .little);
                tokens[j] = reader.buffer[reader.seek..][0..lnd];
                reader.toss(lnd);
            }
        } else {
            try readValue(&reader, valueType);
        }
        //std.debug.print("Key {s}\n", .{key});
    }

    var tensorOffsetList: [256]@Tuple(&[2]type{ []const u8, u64 }) = undefined;

    for (0..tensorCount) |jj| {
        const len = try reader.takeInt(u64, .little);
        const name = try reader.take(len);
        //var buf: [64]u8 = undefined;
        //var stderr = std.debug.lockStderr(&buf);
        //defer std.debug.unlockStderr();
        //try stderr.file_writer.interface.print("Tensor {s}: ", .{name});
        const dimensions = try reader.takeInt(u32, .little);
        var i: usize = 0;
        //(try stderr.file_writer.interface.writableSlice(1))[0] = '[';
        while (i < dimensions) : ({
            i += 1;
        }) {
            const j = try reader.takeInt(u64, .little);
            _ = j;
            //try stderr.file_writer.interface.print("{}", .{j});
            //(try stderr.file_writer.interface.writableSlice(1))[0] = 'x';
        }
        //stderr.file_writer.interface.buffer[stderr.file_writer.interface.end - 1] = ']';
        const tpe = try reader.takeInt(u32, .little);
        _ = tpe;
        //const tpeS = @tagName(@as(Ftype, @enumFromInt(tpe)));
        const offset = try reader.takeInt(u64, .little);

        //try stderr.file_writer.interface.print(": {s} at offset {X}\n", .{ tpeS, offset });
        tensorOffsetList[jj] = .{ name, offset };
    }

    const missing = (alignment - (reader.seek % alignment)) % alignment;
    reader.toss(missing);

    var result = @This(){
        .gguf = data,
        .tensors = @alignCast(data[reader.seek..]),
        .blocks = undefined,
        .outputNormBias = undefined,
        .outputNormWeight = undefined,
        .positionEmbedWeight = undefined,
        .tokenEmdebWeight = undefined,
        .tokens = tokens,
    };

    for (tensorOffsetList[0..tensorCount]) |e| {
        try result.insert(e[0], e[1]);
    }

    return result;
}

fn layerNorm(from: *const Vector, bias: *const Vector, weight: *const Vector, to: *Vector) void {
    @setFloatMode(.optimized);
    const Para = 4;
    var accumulators: [4]@Vector(16, f32) = .{.{0.0} ** 16} ** Para;

    var i: usize = 0;
    while (i < LayerSize) : (i += Para * 16) {
        inline for (0..Para) |j| {
            accumulators[j] += from[i + 16 * j ..][0..16].*;
        }
    }

    const sum = accumulators[0] + accumulators[1] + accumulators[2] + accumulators[3];
    const mid = @reduce(.Add, sum) / LayerSize;
    const vect: @Vector(16, f32) = @splat(mid);

    i = 0;
    accumulators = .{.{0.0} ** 16} ** Para;
    while (i < LayerSize) : (i += Para * 16) {
        inline for (0..Para) |j| {
            accumulators[j] += (from[i + 16 * j ..][0..16].* - vect) * (from[i + 16 * j ..][0..16].* - vect);
        }
    }

    const sum2 = accumulators[0] + accumulators[1] + accumulators[2] + accumulators[3];
    const c = 1.0 / @sqrt(@reduce(.Add, sum2) / LayerSize + 1e-10);
    const cc: @Vector(16, f32) = @splat(c);

    i = 0;
    while (i < LayerSize) : (i += 16) {
        to[i..][0..16].* =
            @mulAdd(@Vector(16, f32), cc * (from[i..][0..16].* - vect), weight[i..][0..16].*, bias[i..][0..16].*);
    }
}

fn convBf16ToF32(from: *const [LayerSize]u16, to: *Vector) void {
    var i: usize = 0;
    while (i < LayerSize) : (i += 16) {
        const f: @Vector(16, u16) = from[i..][0..16].*;

        const widened: @Vector(16, u32) = f;
        const bits: @Vector(16, u32) = widened << @splat(16);
        to[i..][0..16].* = @bitCast(bits);

        //const g: @Vector(1, u16) = .{0};
        //comptime var mask: @Vector(32, i32) = undefined;
        //inline for (0..16) |j| {
        //    mask[2 * j] = -1;
        //    mask[2 * j + 1] = j;
        //}
        //const r: @Vector(32, u16) = @shuffle(u16, f, g, mask);
        //to[i..][0..16].* = @bitCast(r);
    }
}

fn BiasWeightCalc(S: comptime_int, S1: comptime_int, S2: comptime_int, comptime hasBias: bool, comptime hasGelu: bool) type {
    return struct {
        const P = 8;
        //const P = 1;
        bias: if (hasBias) *const [S][S2]f32 else void,
        weight: *const [S][S2][S1]u16,

        optimizedWeight: *const [S][S2 / 16 / P][S1 / 2][P]@Vector(32, u16),

        fn init(alloc: std.mem.Allocator, bias: if (hasBias) *const [S][S2]f32 else void, weight: *const [S][S2][S1]u16) !@This() {
            const optimizedWeight = try createOptimizedWeight(@typeInfo(@FieldType(@This(), "optimizedWeight")).pointer.child, alloc);
            for (0..S) |v| {
                for (0..S2 / 16 / P) |l| {
                    for (0..S1 / 2) |m| {
                        for (0..P) |n| {
                            var r: [32]u16 = undefined;
                            for (0..16) |u| {
                                r[2 * u] = weight[v][16 * P * l + 16 * n + u][2 * m];
                                r[2 * u + 1] = weight[v][16 * P * l + 16 * n + u][2 * m + 1];
                            }
                            optimizedWeight[v][l][m][n] = r;
                        }
                    }
                }
            }
            return .{
                .bias = bias,
                .weight = weight,
                .optimizedWeight = optimizedWeight,
            };
        }

        fn calculateErased(this: *const anyopaque, in: *const anyopaque, outputs: [3]*anyopaque, threadId: usize) void {
            const this_: *const @This() = @ptrCast(@alignCast(this));
            const in_: *const [S1]f32 = @ptrCast(@alignCast(in));
            var outputs_: [S]*[S2]f32 = undefined;
            inline for (0..S) |s| {
                outputs_[s] = @ptrCast(@alignCast(outputs[s]));
            }
            this_.calculate(in_, outputs_, threadId);
        }

        fn calculate(this: *const @This(), in: *const [S1]f32, outputs: [S]*[S2]f32, threadId: usize) void {
            for (outputs, 0..) |o, v| {
                const whole = S2 / 16 / P;
                const begin = whole * threadId / ThreadCount;
                const end = whole * (threadId + 1) / ThreadCount;
                for (begin..end) |l| {
                    var accums: [P]@Vector(16, f32) = undefined;
                    inline for (0..P) |n| {
                        accums[n] = if (hasBias) this.bias[v][16 * P * l + 16 * n ..][0..16].* else @as([16]f32, .{0.0} ** 16);
                    }
                    for (0..S1 / 16) |m0| {
                        const in0: @Vector(16, f32) = in[16 * m0 ..][0..16].*;
                        const inC: @Vector(16, u16) =
                            asm ("vcvtneps2bf16 %[in], %[out]"
                                : [out] "=x" (-> @Vector(16, u16)),
                                : [in] "x" (in0),
                            );

                        var ins: [8]@Vector(32, u16) = undefined;
                        inline for (0..8) |m| {
                            comptime var mask: @Vector(32, usize) = undefined;
                            inline for (0..16) |k| {
                                mask[2 * k] = 2 * m;
                                mask[2 * k + 1] = 2 * m + 1;
                            }
                            ins[m] = @shuffle(u16, inC, inC, mask);
                        }

                        const w1 = this.optimizedWeight[v][l][8 * m0 ..][0..8];
                        const w1p: ?*const [8][8]@Vector(32, u16) =
                            if (8 * m0 + 8 * Prefetch < S1 / 2)
                                this.optimizedWeight[v][l][8 * m0 + 8 * Prefetch ..][0..8]
                            else if (l + 1 < end)
                                this.optimizedWeight[v][l + 1][8 * m0 + 8 * Prefetch - S1 / 2 ..][0..8]
                            else
                                null;
                        for (0..8) |m| {
                            const in1 = ins[m];
                            const w2 = w1[m];
                            if (comptime Prefetch > 0) {
                                if (w1p) |w| {
                                    inline for (0..P) |n| {
                                        @prefetch(&w[m][n], .{ .locality = 0 });
                                    }
                                }
                            }
                            inline for (0..P) |n| {
                                const weights: @Vector(32, u16) = w2[n];
                                accums[n] = asm ("vdpbf16ps %[in1], %[in2], %[acc]"
                                    : [acc] "=v" (-> @Vector(16, f32)),
                                    : [acc_in] "0" (accums[n]),
                                      [in1] "v" (weights),
                                      [in2] "v" (in1),
                                );
                            }
                        }
                    }
                    for (0..P) |n| {
                        const result: [16]f32 = if (hasGelu) @import("gelu.zig").gelu(accums[n]) else accums[n];
                        o[16 * P * l + 16 * n ..][0..16].* = result;
                    }
                }
            }
        }
    };
}

const KvCache = struct {
    const MaxLen = 1024;

    weights: BiasWeightCalc(3, LayerSize, LayerSize, true, false),

    ks: *[MaxLen / 16][LayerSize][16]f32,
    vs: *[MaxLen]Vector,
    length: usize,

    fn init(alloc: std.mem.Allocator, attnQkvBias: *const [3]Vector, attnQkvWeight: *const [3][LayerSize][LayerSize]u16) !KvCache {
        const ks = try alloc.create([MaxLen / 16][LayerSize][16]f32);
        for (ks) |*k| {
            for (k) |*kk| {
                for (0..16) |i| {
                    kk[i] = 0.0;
                }
            }
        }
        const vs = try alloc.create([MaxLen]Vector);
        for (vs) |*v| {
            for (v) |*vv| {
                vv.* = -std.math.inf(f32);
            }
        }

        return .{
            .weights = try BiasWeightCalc(3, LayerSize, LayerSize, true, false).init(alloc, attnQkvBias, attnQkvWeight),
            .ks = ks,
            .vs = vs,
            .length = 0,
        };
    }

    fn next(this: *@This(), syncThreads: *SyncThreads, in: *const Vector, q: *Vector) void {
        var k: Vector = undefined;
        const outputs: [3]*Vector = .{ q, &k, &this.vs[this.length] };
        syncThreads.calculate(3, LayerSize, LayerSize, true, false, &this.weights, in, outputs);
        for (0..LayerSize) |i| {
            this.ks[this.length / 16][i][this.length % 16] = k[i];
        }
        this.length += 1;
    }

    fn attend(this: *const @This(), q: *const Vector, out: *Vector) void {
        @setFloatMode(.optimized);
        for (0..Heads) |h| {
            var s: [1024]f32 = undefined;
            var smax: @Vector(16, f32) = @splat(-std.math.inf(f32));
            const lenUp = (this.length + 15) / 16;
            for (0..lenUp) |i| {
                var acc: @Vector(16, f32) = @splat(0.0);
                for (0..64) |w| {
                    const qq: @Vector(16, f32) = @splat(q[h * 64 + w]);
                    acc += qq * this.ks[i][h * 64 + w];
                }
                acc *= @splat(0.125);
                inline for (0..16) |lane| {
                    if (i + 1 == lenUp and this.length % 16 != 0 and this.length % 16 <= lane) {
                        acc[lane] = -std.math.inf(f32);
                    }
                }
                s[16 * i ..][0..16].* = acc;
                smax = @max(smax, acc);
            }
            const max = @reduce(.Max, smax);
            var sum: f32 = 0.0;
            for (0..16 * lenUp) |i| {
                s[i] = @exp(s[i] - max);
                sum += s[i];
            }
            const sumR = 1.0 / sum;
            const sumRV: @Vector(16, f32) = @splat(sumR);
            for (0..64) |w| {
                out[h * 64 + w] = 0.0;
            }
            for (0..this.length) |i| {
                for (0..64 / 16) |w| {
                    const sI: @Vector(16, f32) = @splat(s[i]);
                    const vec: @Vector(16, f32) = out[h * 64 + w * 16 ..][0..16].*;
                    out[h * 64 + w * 16 ..][0..16].* = vec + sI * sumRV * this.vs[i][h * 64 + w * 16 ..][0..16].*;
                }
            }
        }
    }
};

const LayerCalculation = struct {
    normBias: *const Vector,
    normWeight: *const Vector,
    ffnBias: *const Vector,
    ffnWeight: *const Vector,

    kvCache: KvCache,

    output: BiasWeightCalc(1, LayerSize, LayerSize, true, false),

    ffnUp: BiasWeightCalc(1, LayerSize, Hidden, true, true),
    ffnDown: BiasWeightCalc(1, Hidden, LayerSize, true, false),

    fn attend(this: *@This(), syncThreads: *SyncThreads, in: *const Vector, out: *Vector) void {
        var v: Vector = undefined;
        layerNorm(in, this.normBias, this.normWeight, &v);

        var q: Vector = undefined;
        var a: Vector = undefined;

        this.kvCache.next(syncThreads, &v, &q);
        this.kvCache.attend(&q, &a);

        syncThreads.calculate(1, LayerSize, LayerSize, true, false, &this.output, &a, .{out});

        for (0..LayerSize) |i| {
            out[i] += in[i];
        }
    }

    fn ffn(this: *@This(), syncThreads: *SyncThreads, in: *const Vector, out: *Vector) void {
        var v: Vector = undefined;
        var hidden: [Hidden]f32 = undefined;
        layerNorm(in, this.ffnBias, this.ffnWeight, &v);
        syncThreads.calculate(1, LayerSize, Hidden, true, true, &this.ffnUp, &v, .{&hidden});
        syncThreads.calculate(1, Hidden, LayerSize, true, false, &this.ffnDown, &hidden, .{out});
        for (0..LayerSize) |i| {
            out[i] += in[i];
        }
    }
};

const LogitsF = struct {
    calc: BiasWeightCalc(1, LayerSize, 50304, false, false),

    fn init(alloc: std.mem.Allocator, tokenWeights: *const [TokenCount][LayerSize]u16) !LogitsF {
        const tmp = try alloc.create([50304][LayerSize]u16);
        @memmove(tmp[0..TokenCount], tokenWeights);
        for (TokenCount..50304) |i| {
            for (0..LayerSize) |j| {
                tmp[i][j] = 0.0;
            }
        }
        return .{ .calc = try BiasWeightCalc(1, LayerSize, 50304, false, false).init(alloc, {}, tmp) };
    }

    fn logits(this: *const @This(), syncThreads: *SyncThreads, v: *const Vector, l: *[50304]f32) void {
        syncThreads.calculate(1, LayerSize, 50304, false, false, &this.calc, v, .{l});
    }
};

// Keep the four most likely tokens and renormalize their softmax probabilities.
// draw is uniform in [0, 1); accepting it explicitly makes sampling testable.
fn sampleTopFour(logits: []const f32, draw: f64) u16 {
    std.debug.assert(logits.len >= 4);
    std.debug.assert(draw >= 0.0 and draw < 1.0);
    var top: [4]f32 = .{-std.math.inf(f32)} ** 4;
    var indices: [4]u16 = .{0} ** 4;
    for (logits, 0..) |logit, i| {
        if (logit <= top[3]) continue;
        var slot: usize = 3;
        while (slot > 0 and logit > top[slot - 1]) : (slot -= 1) {
            top[slot] = top[slot - 1];
            indices[slot] = indices[slot - 1];
        }
        top[slot] = logit;
        indices[slot] = @intCast(i);
    }

    var weights: [4]f64 = undefined;
    var total: f64 = 0.0;
    for (top, &weights) |logit, *weight| {
        weight.* = @exp(@as(f64, logit) - @as(f64, top[0]));
        total += weight.*;
    }
    const threshold = draw * total;
    var cumulative: f64 = 0.0;
    for (weights, indices) |weight, index| {
        cumulative += weight;
        if (threshold < cumulative) return index;
    }
    return indices[3]; // Guard against rounding at the upper boundary.
}

fn logitsF(logF: *const LogitsF, syncThreads: *SyncThreads, v: *const Vector, logits: *[50304]f32, tokens: *const [TokenCount][]const u8, random: std.Random) u16 {
    logF.logits(syncThreads, v, logits);
    const next = sampleTopFour(logits[0..TokenCount], random.float(f64));
    std.debug.print("{f}", .{DecodedToken{ .token = tokens[next] }});
    return next;
}

const SyncThreads = struct {
    currentTask: usize,
    calculateFn: *const fn (
        *const anyopaque,
        *const anyopaque,
        [3]*anyopaque,
        usize,
    ) void,
    calculateData: *const anyopaque,
    calculateIn: *const anyopaque,
    calculateOut: [3]*anyopaque,
    stp: bool,
    threads: [ThreadCount - 1]?std.Thread,
    finished: [ThreadCount - 1]u512 align(64), // align on cache line

    fn init() SyncThreads {
        return .{
            .currentTask = 0,
            .calculateData = undefined,
            .calculateIn = undefined,
            .calculateOut = undefined,
            .calculateFn = undefined,
            .finished = .{0} ** (ThreadCount - 1),
            .stp = false,
            .threads = .{null} ** (ThreadCount - 1),
        };
    }

    fn start(this: *@This()) !void {
        for (0..ThreadCount - 1) |i| {
            this.threads[i] = try std.Thread.spawn(.{}, SyncThreads.run, .{ this, i });
        }
    }

    fn run(this: *@This(), threadId: usize) void {
        var nextTask: usize = 1;
        while (true) {
            while (@atomicLoad(usize, &this.currentTask, .acquire) != nextTask) {
                std.atomic.spinLoopHint();
            }
            if (this.stp) return;
            this.calculateFn(this.calculateData, this.calculateIn, this.calculateOut, threadId);
            @atomicStore(usize, @as(*usize, @ptrCast(&this.finished[threadId])), nextTask, .release);
            nextTask += 1;
        }
    }

    fn calculate(this: *@This(), S: comptime_int, S1: comptime_int, S2: comptime_int, comptime hasBias: bool, comptime hasGelu: bool, calc: *const BiasWeightCalc(S, S1, S2, hasBias, hasGelu), in: *const [S1]f32, outputs: [S]*[S2]f32) void {
        this.calculateFn = BiasWeightCalc(S, S1, S2, hasBias, hasGelu).calculateErased;
        this.calculateData = calc;
        this.calculateIn = in;
        inline for (0..S) |s| {
            this.calculateOut[s] = outputs[s];
        }
        const ct = this.currentTask + 1;
        @atomicStore(usize, &this.currentTask, ct, .release);
        calc.calculate(in, outputs, ThreadCount - 1);
        w: while (true) {
            for (0..ThreadCount - 1) |i| {
                if (@atomicLoad(usize, @as(*usize, @ptrCast(&this.finished[i])), .acquire) != ct) {
                    std.atomic.spinLoopHint();
                    continue :w;
                }
            }
            return;
        }
    }

    fn stop(this: *@This()) void {
        this.stp = true;
        @atomicStore(usize, &this.currentTask, this.currentTask + 1, .release);
        for (this.threads) |t| {
            t.?.join();
        }
    }
};

pub fn main(ini: std.process.Init) !void {
    var iter = ini.minimal.args.iterate();
    _ = iter.skip();
    return run(ini, iter);
}

pub fn run(ini: std.process.Init, args: anytype) !void {
    var iter = args;
    const filename = iter.next() orelse "";
    const vv = try init(ini.io, filename, ini.arena.allocator());
    var ids: std.ArrayList(u16) = .empty;
    while (iter.next()) |arg| {
        try ids.append(ini.arena.allocator(), try std.fmt.parseInt(u16, arg, 10));
    }
    return generate(ini, &vv, ids.items, false);
}

pub fn runText(ini: std.process.Init, args: anytype, comptime tokenizer: type) !void {
    var iter = args;
    const filename = iter.next() orelse return error.MissingGgufPath;
    const prompt = if (iter.next()) |text| text else blk: {
        var buffer: [4096]u8 = undefined;
        var reader = std.Io.File.stdin().reader(ini.io, &buffer);
        break :blk try reader.interface.allocRemaining(ini.arena.allocator(), .unlimited);
    };
    if (iter.next() != null) return error.TooManyArguments;
    if (prompt.len == 0) return error.EmptyPrompt;
    const ids = try tokenizer.encodeTableOrGenerate(ini.io, ini.arena.allocator(), filename, "tokens", prompt);
    const vv = try init(ini.io, filename, ini.arena.allocator());
    return generate(ini, &vv, ids, true);
}

fn generate(ini: std.process.Init, vv: *const @This(), ids: []const u16, text_output: bool) !void {
    if (ids.len > 1024) return error.PromptTooLong;
    if (text_output and ids.len == 1024) return error.PromptTooLong;
    for (ids) |id| if (id >= TokenCount) return error.InvalidTokenId;
    var layers: [12]LayerCalculation = undefined;
    for (0..12) |i| {
        layers[i] = .{
            .normBias = vv.blocks[i].attnNormBias,
            .normWeight = vv.blocks[i].attnNormWeight,
            .ffnBias = vv.blocks[i].ffnNormBias,
            .ffnWeight = vv.blocks[i].ffnNormWeight,
            .kvCache = try KvCache.init(ini.arena.allocator(), vv.blocks[i].attnQkvBias, vv.blocks[i].attnQkvWeight),
            .output = try BiasWeightCalc(1, LayerSize, LayerSize, true, false).init(ini.arena.allocator(), vv.blocks[i].attnOutputBias, vv.blocks[i].attnOutputWeight),
            .ffnUp = try BiasWeightCalc(1, LayerSize, Hidden, true, true).init(ini.arena.allocator(), vv.blocks[i].ffnUpBias, vv.blocks[i].ffnUpWeight),
            .ffnDown = try BiasWeightCalc(1, Hidden, LayerSize, true, false).init(ini.arena.allocator(), vv.blocks[i].ffnDownBias, vv.blocks[i].ffnDownWeight),
        };
    }

    var threadSync = SyncThreads.init();
    try threadSync.start();
    defer threadSync.stop();

    var seed: u64 = undefined;
    std.Io.random(ini.io, std.mem.asBytes(&seed));
    var prng = std.Random.DefaultPrng.init(seed);

    var pos: usize = 0;
    var vec: Vector = undefined;
    for (ids) |j| {
        const token = &vv.tokenEmdebWeight[j];
        if (text_output) {
            std.debug.print("{f}", .{DecodedToken{ .token = vv.tokens[j] }});
        } else {
            std.debug.print("{s}", .{vv.tokens[j]});
        }
        convBf16ToF32(token, &vec);
        for (0..LayerSize) |u| {
            vec[u] += vv.positionEmbedWeight[pos][u];
        }
        pos += 1;

        for (0..12) |layer| {
            var v: Vector = undefined;
            layers[layer].attend(&threadSync, &vec, &v);
            layers[layer].ffn(&threadSync, &v, &vec);
        }
    }
    if (!text_output) std.debug.print("\n", .{});

    const lg = try LogitsF.init(ini.arena.allocator(), vv.tokenEmdebWeight);

    if (pos > 0) {
        const limit = if (text_output) @min(ids.len + 200, 1024) else 200;
        while (pos < limit) {
            var to: Vector = undefined;
            var logits: [50304]f32 = undefined;
            layerNorm(&vec, vv.outputNormBias, vv.outputNormWeight, &to);
            const next = logitsF(&lg, &threadSync, &to, &logits, vv.tokens, prng.random());
            convBf16ToF32(&vv.tokenEmdebWeight[next], &vec);
            for (0..LayerSize) |u| {
                vec[u] += vv.positionEmbedWeight[pos][u];
            }
            pos += 1;

            for (0..12) |layer| {
                var v: Vector = undefined;
                layers[layer].attend(&threadSync, &vec, &v);
                layers[layer].ffn(&threadSync, &v, &vec);
            }
        }
    }
    if (text_output) std.debug.print("\n", .{});
}

fn gpt2CodepointToByte(cp: u21) ?u8 {
    // Bytes kept unchanged by GPT-2's byte encoder.
    if (cp >= '!' and cp <= '~') return @intCast(cp);
    if (cp >= 0x00A1 and cp <= 0x00AC) return @intCast(cp);
    if (cp >= 0x00AE and cp <= 0x00FF) return @intCast(cp);

    // Bytes represented by synthetic Unicode code points.
    if (cp >= 0x0100 and cp <= 0x0120)
        return @intCast(cp - 0x0100); // bytes 0–32

    if (cp >= 0x0121 and cp <= 0x0142)
        return @intCast(cp - 0x0121 + 127); // bytes 127–160

    if (cp == 0x0143) return 173;

    return null;
}

const DecodedToken = struct {
    token: []const u8,

    pub fn format(self: @This(), writer: *std.Io.Writer) !void {
        var view = std.unicode.Utf8View.init(self.token) catch {
            return error.WriteFailed;
        };
        var iterator = view.iterator();

        while (iterator.nextCodepoint()) |cp| {
            if (gpt2CodepointToByte(cp)) |byte| {
                try writer.writeByte(byte);
            } else {
                var buf: [4]u8 = undefined;
                const len = std.unicode.utf8Encode(cp, &buf) catch {
                    return error.WriteFailed;
                };
                try writer.writeAll(buf[0..len]);
            }
        }
    }
};

test {
    _ = @import("gelu.zig");
}

test "order in simd" {
    const a: [16]u16 = .{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    const b: @Vector(16, u16) = a;
    inline for (0..16) |i| {
        try std.testing.expectEqual(a[i], b[i]);
    }
}

test "top four sampling preserves relative probabilities and excludes other tokens" {
    // Unsorted weights 1, 4, 2, 3, plus a fifth candidate that must be excluded.
    const logits = [_]f32{ 0.0, @log(@as(f32, 4.0)), @log(@as(f32, 2.0)), @log(@as(f32, 3.0)), -0.1 };
    var counts: [5]usize = .{0} ** 5;
    for (0..10000) |i| {
        const draw = (@as(f64, @floatFromInt(i)) + 0.5) / 10000.0;
        counts[sampleTopFour(&logits, draw)] += 1;
    }
    try std.testing.expectEqualSlices(usize, &.{ 1000, 4000, 2000, 3000, 0 }, &counts);
}

test "top four sampling handles ties and extreme logit gaps" {
    const tied = [_]f32{ 5.0, 5.0, 5.0, 5.0, 5.0 };
    for (0..4) |i| {
        try std.testing.expectEqual(@as(u16, @intCast(i)), sampleTopFour(&tied, @as(f64, @floatFromInt(i)) / 4.0));
    }
    const extreme = [_]f32{ -10000.0, 10000.0, -20000.0, 0.0, -30000.0 };
    try std.testing.expectEqual(@as(u16, 1), sampleTopFour(&extreme, 0.0));
    try std.testing.expectEqual(@as(u16, 1), sampleTopFour(&extreme, 0.9999999999999999));
}
