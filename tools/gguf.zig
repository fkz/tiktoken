const std = @import("std");
const BlockCount = 12;
const LayerSize = 768;
const Hidden = 4 * LayerSize;
const Vector = [LayerSize]f32;
const Matrix = [LayerSize][LayerSize]u16;
const TokenCount = 50257;
const Heads = 12;

const BlockData = struct {
    attnNormBias: *const Vector,
    attnNormWeight: *const Vector,
    attnQkvBias: *const [3]Vector,
    attnQkvWeight: *const [3][LayerSize][LayerSize]u16,
    attnOutputBias: *const Vector,
    attnOutputWeight: *const Matrix,
    ffnNormBias: *const Vector,
    ffnNormWeight: *const Vector,
    ffnUpBias: *const [Hidden]f32,
    ffnUpWeight: *const [Hidden][LayerSize]u16,
    ffnDownBias: *const Vector,
    ffnDownWeight: *const [LayerSize][Hidden]u16,
};

gguf: []align(std.heap.page_size_min) u8,

tensors: []align(32) u8,

blocks: [BlockCount]BlockData,
outputNormBias: *const Vector,
outputNormWeight: *const Vector,
positionEmbedWeight: *const [1024]Vector,
tokenEmdebWeight: *const [TokenCount][LayerSize]u16,

tokens: *const [TokenCount][]const u8,

fn tensorAt(this: *@This(), T: type, offset: u64) *const T {
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
        3 => reader.toss(3),
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
        std.debug.print("Key {s}\n", .{key});
    }

    var tensorOffsetList: [256]@Tuple(&[2]type{ []const u8, u64 }) = undefined;

    for (0..tensorCount) |jj| {
        const len = try reader.takeInt(u64, .little);
        const name = try reader.take(len);
        var buf: [64]u8 = undefined;
        var stderr = std.debug.lockStderr(&buf);
        defer std.debug.unlockStderr();
        try stderr.file_writer.interface.print("Tensor {s}: ", .{name});
        const dimensions = try reader.takeInt(u32, .little);
        var i: usize = 0;
        (try stderr.file_writer.interface.writableSlice(1))[0] = '[';
        while (i < dimensions) : ({
            i += 1;
        }) {
            const j = try reader.takeInt(u64, .little);
            try stderr.file_writer.interface.print("{}", .{j});
            (try stderr.file_writer.interface.writableSlice(1))[0] = 'x';
        }
        stderr.file_writer.interface.buffer[stderr.file_writer.interface.end - 1] = ']';
        const tpe = try reader.takeInt(u32, .little);
        const tpeS = @tagName(@as(Ftype, @enumFromInt(tpe)));
        const offset = try reader.takeInt(u64, .little);

        std.debug.print(": {s} at offset {X}\n", .{ tpeS, offset });
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
            accumulators[j] += (from[i..][0..16].* - vect) * (from[i..][0..16].* - vect);
        }
    }

    const sum2 = accumulators[0] + accumulators[1] + accumulators[2] + accumulators[3];
    const c = 1.0 / @sqrt(@reduce(.Add, sum2) + 1e-10);
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

fn BiasWeightCalc(S: comptime_int, S1: comptime_int, S2: comptime_int) type {
    return struct {
        bias: *const [S][S2]f32,
        weight: *const [S][S2][S1]u16,

        optimizedWeight: *const [S][S2 / 16 / 8][S1 / 2][8]@Vector(32, u16),

        fn init(alloc: std.mem.Allocator, bias: *const [S][S2]f32, weight: *const [S][S2][S1]u16) !@This() {
            const optimizedWeight = try alloc.create(@typeInfo(@FieldType(@This(), "optimizedWeight")).pointer.child);
            for (0..S) |v| {
                for (0..S2 / 16 / 8) |l| {
                    for (0..S1 / 2) |m| {
                        for (0..8) |n| {
                            var r: [32]u16 = undefined;
                            for (0..16) |u| {
                                r[2 * u] = weight[v][16 * 8 * l + 16 * n + u][2 * m];
                                r[2 * u + 1] = weight[v][16 * 8 * l + 16 * n + u][2 * m + 1];
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

        fn calculate(this: *const @This(), in: *const [S1]f32, outputs: [S]*[S2]f32) void {
            for (outputs, 0..) |o, j| {
                for (0..S2) |i| {
                    o[i] = this.bias[j][i];
                }
            }
            for (outputs, 0..) |o, v| {
                for (0..S2 / 16 / 8) |l| {
                    var accums: [8]@Vector(16, f32) = undefined;
                    inline for (0..8) |n| {
                        inline for (0..16) |u| {
                            accums[n][u] = this.bias[v][16 * 8 * l + 16 * n + u];
                        }
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
                            inline for (0..8) |k| {
                                mask[2 * k] = 2 * m;
                                mask[2 * k + 1] = 2 * m + 1;
                            }
                            ins[m] = @shuffle(u16, inC, inC, mask);
                        }

                        for (0..8) |m| {
                            const in1 = ins[m];
                            inline for (0..8) |n| {
                                const weights: @Vector(32, u16) = this.optimizedWeight[v][l][m][n];
                                accums[n] = asm ("vdpbf16ps %[in1], %[in2], %[acc]"
                                    : [acc] "=v" (-> @Vector(16, f32)),
                                    : [acc_in] "0" (accums[n]),
                                      [in1] "v" (weights),
                                      [in2] "v" (in1),
                                );
                            }
                        }
                    }
                    for (0..8) |n| {
                        o[16 * 8 * l + 16 * n ..][0..16].* = accums[n];
                    }
                }
            }
        }
    };
}

const KvCache = struct {
    weights: BiasWeightCalc(3, LayerSize, LayerSize),

    ks: []Vector,
    vs: []Vector,
    length: usize,

    fn init(alloc: std.mem.Allocator, attnQkvBias: *const [3]Vector, attnQkvWeight: *const [3][LayerSize][LayerSize]u16) !KvCache {
        return .{
            .weights = try BiasWeightCalc(3, LayerSize, LayerSize).init(alloc, attnQkvBias, attnQkvWeight),
            .ks = try alloc.alloc(Vector, 1024),
            .vs = try alloc.alloc(Vector, 1024),
            .length = 0,
        };
    }

    fn next(this: *@This(), in: *const Vector, q: *Vector) void {
        const outputs: [3]*Vector = .{ q, &this.ks[this.length], &this.vs[this.length] };
        this.length += 1;
        this.weights.calculate(in, outputs);
    }

    fn attend(this: *const @This(), q: *const Vector, out: *Vector) void {
        for (0..Heads) |h| {
            var s: [1024]f32 = undefined;
            var smax = -std.math.inf(f32);
            for (0..this.length) |i| {
                s[i] = 0.0;
                for (0..64) |w| {
                    s[i] += q[h * 64 + w] * this.ks[i][h * 64 + w];
                }
                s[i] *= 0.125;
                if (smax < s[i]) smax = s[i];
            }
            var sum: f32 = 0.0;
            for (0..this.length) |i| {
                s[i] = @exp(s[i] - smax);
                sum += s[i];
            }
            const sumR = 1.0 / sum;
            for (0..this.length) |i| {
                for (0..64) |w| {
                    out[h * 64 + w] = s[i] * sumR * this.vs[i][h * 64 + w];
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

    output: BiasWeightCalc(1, LayerSize, LayerSize),

    ffnUp: BiasWeightCalc(1, LayerSize, Hidden),
    ffnDown: BiasWeightCalc(1, Hidden, LayerSize),

    fn attend(this: *@This(), in: *const Vector, out: *Vector) void {
        var v: Vector = undefined;
        layerNorm(in, this.normBias, this.normWeight, &v);

        var q: Vector = undefined;
        var a: Vector = undefined;

        this.kvCache.next(&v, &q);
        this.kvCache.attend(&q, &a);
        this.output.calculate(&a, .{out});

        for (0..LayerSize) |i| {
            out[i] += in[i];
        }
    }

    const u = @sqrt(2.0 / std.math.pi);

    fn gelu(v: *[Hidden]f32) void {
        for (0..Hidden) |i| {
            const c = v[i];
            const r = c / 2.0 * (1.0 + std.math.tanh(u * (c + 0.044715 * c * c * c)));
            v[i] = r;
        }
    }

    fn ffn(this: *@This(), in: *const Vector, out: *Vector) void {
        var v: Vector = undefined;
        var hidden: [Hidden]f32 = undefined;
        layerNorm(in, this.ffnBias, this.ffnWeight, &v);
        this.ffnUp.calculate(&v, .{&hidden});
        gelu(&hidden);
        this.ffnDown.calculate(&hidden, .{out});
        for (0..LayerSize) |i| {
            out[i] += in[i];
        }
    }
};

fn logitsF(v: *const Vector, tokenWeights: *const [TokenCount][LayerSize]u16, logits: *[TokenCount]f32, tokens: *const [TokenCount][]const u8) void {
    for (0..TokenCount) |i| {
        logits[i] = 0.0;
    }

    for (0..LayerSize) |l| {
        for (0..TokenCount) |i| {
            logits[i] += v[l] * tokenWeights[i][l];
        }
    }

    var maxim: [8]f32 = .{0.0} ** 8;
    var maximIndex: [8]u16 = .{0} ** 8;

    for (0..TokenCount) |i| {
        if (logits[i] >= maxim[7]) {
            maxim[7] = logits[i];
            maximIndex[7] = @intCast(i);
            for (1..8) |m| {
                if (maxim[7 - m] < logits[i]) {
                    maxim[8 - m] = maxim[7 - m];
                    maxim[7 - m] = logits[i];
                    maximIndex[8 - m] = maximIndex[7 - m];
                    maximIndex[7 - m] = @intCast(i);
                } else {
                    break;
                }
            }
        }
    }

    var sum: f32 = 0.0;
    const max = maxim[0];
    std.debug.print("max {}\n", .{max});
    for (0..TokenCount) |i| {
        logits[i] = @exp(logits[i] - max);
        sum += logits[i];
    }

    for (0..8) |i| {
        std.debug.print("token {} '{s}' prob {d:.6} {}\n", .{ maximIndex[i], tokens[maximIndex[i]], logits[maximIndex[i]] / sum, maxim[i] });
    }
}

pub fn main(ini: std.process.Init) !void {
    var iter = ini.minimal.args.iterate();
    _ = iter.skip();
    const filename = iter.next() orelse "";
    const vv = try init(ini.io, filename, ini.arena.allocator());

    var layers: [12]LayerCalculation = undefined;
    for (0..12) |i| {
        layers[i] = .{
            .normBias = vv.blocks[i].attnNormBias,
            .normWeight = vv.blocks[i].attnNormWeight,
            .ffnBias = vv.blocks[i].ffnNormBias,
            .ffnWeight = vv.blocks[i].ffnNormWeight,
            .kvCache = try KvCache.init(ini.arena.allocator(), vv.blocks[i].attnQkvBias, vv.blocks[i].attnQkvWeight),
            .output = try BiasWeightCalc(1, LayerSize, LayerSize).init(ini.arena.allocator(), vv.blocks[i].attnOutputBias, vv.blocks[i].attnOutputWeight),
            .ffnUp = try BiasWeightCalc(1, LayerSize, Hidden).init(ini.arena.allocator(), vv.blocks[i].ffnUpBias, vv.blocks[i].ffnUpWeight),
            .ffnDown = try BiasWeightCalc(1, Hidden, LayerSize).init(ini.arena.allocator(), vv.blocks[i].ffnDownBias, vv.blocks[i].ffnDownWeight),
        };
    }

    var pos: usize = 0;
    var vec: Vector = undefined;
    while (iter.next()) |a| {
        const j = try std.fmt.parseInt(u16, a, 10);
        const token = &vv.tokenEmdebWeight[j];
        std.debug.print("{s}", .{vv.tokens[j]});
        convBf16ToF32(token, &vec);
        for (0..LayerSize) |u| {
            vec[u] += vv.positionEmbedWeight[pos][u];
        }
        pos += 1;

        for (0..12) |layer| {
            var v: Vector = undefined;
            layers[layer].attend(&vec, &v);
            layers[layer].ffn(&v, &vec);
        }
    }
    std.debug.print("\n", .{});

    if (pos > 0) {
        var to: Vector = undefined;
        var logits: [TokenCount]f32 = undefined;
        layerNorm(&vec, vv.outputNormBias, vv.outputNormWeight, &to);
        logitsF(&to, vv.tokenEmdebWeight, &logits, vv.tokens);
    }
}

test "order in simd" {
    const a: [16]u16 = .{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    const b: @Vector(16, u16) = a;
    inline for (0..16) |i| {
        try std.testing.expectEqual(a[i], b[i]);
    }
}
