const std = @import("std");

const TokenValue = struct {
    token1: u16 = 0,
    token2: u16 = 0,
    resultToken: u16 = 0,
};

const Size = 50000;

const Bits = 14;
const CacheSize = std.math.pow(usize, 2, Bits);

fn findTokenId(strs: *const [Size][2][]const u8, str: []const u8, count: u16) ?u16 {
    if (str.len == 1) return str[0];
    if (str.len == 2 and str[0] & 0xFE == 0xC2) {
        return 128 + (str[0] & 1) * 64 + (str[1] & 63);
    }
    if (str.len == 2 and str[0] & 0xFE == 0xC4) {
        const v: u8 = (str[0] & 1) * 64 + str[1] & 63;
        if (v <= 32) {
            return v;
        }
        if (v <= 66) {
            return v + 0x5E;
        }
        if (v == 67) {
            return 0xAD;
        }
    }
    for (256..count) |i| {
        const t1 = strs[i - 256][0];
        const t2 = strs[i - 256][1];
        if (t1.len + t2.len != str.len) {
            continue;
        }
        if (std.mem.eql(u8, t1, str[0..t1.len]) and
            std.mem.eql(u8, t2, str[t1.len..]))
        {
            return @intCast(i);
        }
    }
    return null;
}

fn tokens(merges: []const u8) ?[Size]TokenValue {
    var it = std.mem.splitScalar(u8, merges, '\n');
    return tokensFromIterator(&it);
}

fn tokensFromIterator(it: anytype) ?[Size]TokenValue {
    var index: u16 = 256;
    var strs: [Size][2][]const u8 = undefined;
    var result: [Size]TokenValue = .{TokenValue{}} ** Size;
    var progress: u16 = 0;
    while (it.next()) |line| {
        if (line.len == 0) continue;
        if (index - 256 >= Size) return null;
        var l = std.mem.splitScalar(u8, line, ' ');
        const t1 = l.next() orelse return null;
        const t2 = l.next() orelse return null;
        if (t1.len == 0 or t2.len == 0 or l.next() != null) return null;
        const t1tok = findTokenId(&strs, t1, index) orelse return null;
        const t2tok = findTokenId(&strs, t2, index) orelse return null;
        strs[index - 256] = .{ t1, t2 };
        result[index - 256] = TokenValue{
            .token1 = t1tok,
            .token2 = t2tok,
            .resultToken = index,
        };
        index += 1;
        const newProgress = @as(usize, index - 256) * 4 / Size * 25;
        if (progress < newProgress) {
            progress = @intCast(newProgress);
            std.debug.print("Progress: {}%\n", .{progress});
        }
    }
    return result;
}

fn hash(t1: u16, t2: u16, hv: u32) u32 {
    const t2s: u32 = t2;
    const result: u32 = (t2s << 16) | t1;
    return (result *% hv) >> (32 - Bits);
}

const lookupTable: [512]u8 = blk: {
    var result: [512]u8 = undefined;
    var index = 0;
    while (index < 512) {
        const value = index >> 1;
        if (index & 1 == 1 and value & 0xF0 != 0xF0) {
            result[index] = value + 16;
        } else if (value & 0x0F != 0x0F) {
            result[index] = value + 1;
        } else {
            result[index] = value;
        }
        index += 1;
    }
    break :blk result;
};

fn loop1(toks: *const [Size]TokenValue, hv: u32, counts: *[CacheSize]u8, breakCond: u8) bool {
    for (toks) |t| {
        if (t.resultToken == 0) continue;
        const h = hash(t.token1, t.token2, hv);
        counts[h] +|= 1;
        if (counts[h] == breakCond) return false;
    }
    return true;
}

fn loop2(counts: *const [CacheSize]u8) [4]u16 {
    var in: usize = 0;
    var result: u8 = 0;
    var count: u16 = 0;
    var count2: u16 = 0;
    while (in < CacheSize) {
        const values: @Vector(32, u8) = counts[in..][0..32].*;
        const max = @reduce(.Max, values);

        if (max > result) {
            @branchHint(.cold);
            const u = @reduce(.Add, @select(u8, values == @as(@Vector(32, u8), @splat(max - 1)), @as(@Vector(32, u8), @splat(1)), @as(@Vector(32, u8), @splat(0))));
            if (max == result + 1) {
                count2 = count + u;
            } else {
                count2 = u;
            }
            count = @reduce(.Add, @select(u8, values == @as(@Vector(32, u8), @splat(max)), @as(@Vector(32, u8), @splat(1)), @as(@Vector(32, u8), @splat(0))));
            result = max;
        } else if (max == result or max == result - 1) {
            @branchHint(.unlikely);
            count += @reduce(.Add, @select(u8, values == @as(@Vector(32, u8), @splat(result)), @as(@Vector(32, u8), @splat(1)), @as(@Vector(32, u8), @splat(0))));
            count2 += @reduce(.Add, @select(u8, values == @as(@Vector(32, u8), @splat(result - 1)), @as(@Vector(32, u8), @splat(1)), @as(@Vector(32, u8), @splat(0))));
        }
        in += 32;
    }
    return .{ result, count, count2, 0 };
}

fn init0(cache: *[CacheSize]u8) void {
    asm volatile (
        \\xor %%eax, %%eax
        \\rep stosb
        :
        : [dst] "{rdi}" (cache),
          [len] "{rcx}" (CacheSize),
        : .{ .rax = true, .rdi = true, .rcx = true, .memory = true });
}

fn hashTok(toks: *[Size]TokenValue, hv: u32, breakCond: u8) [4]u16 {
    var counts: [CacheSize]u8 = undefined;
    init0(&counts);
    if (loop1(toks, hv, &counts, breakCond)) {
        //return linearlySpreadHashes(&counts);
        return loop2(&counts);
        //result[3] = result[2];
        //result[2] = result[1];
        //result[1] = lsh;
        //return result;
    } else {
        return .{ breakCond * 2, 0xFF, 0xFF, 0xFF };
    }
}

fn tryHashTok(io: std.Io, toks: *[Size]TokenValue, count: usize, initBreakCond: u8) !void {
    var c: usize = 0;
    var over9: usize = 0xFFFF;
    const breakCond = initBreakCond;
    while (c < count) {
        var buffer: [4]u8 = undefined;
        std.Io.random(io, &buffer);
        const r = hashTok(toks, std.mem.readInt(u32, &buffer, .native), breakCond);
        var over9new: usize = undefined;
        if (r[0] > 9) {
            over9new = r[0] * r[1] + (r[0] - 1) * r[2];
        } else if (r[0] == 9) {
            over9new = 9 * r[1];
        } else {
            over9new = r[0];
        }

        //if (r[0] < max[0] or r[0] == max[0] and r[1] < max[1] or r[0] == max[0] and r[1] == max[1] and r[2] < max[2]) {
        if (over9new < over9) {
            over9 = over9new;
            //max = r;
            //breakCond = @min(breakCond, max[0] + 1);
            std.debug.print("New reduction to {} ({},{},{}) in step {} using {X}\n", .{ r[0], r[1], r[2], r[3], c, buffer });
        } else if (c % 1024 == 0) {
            const pprev = (c - 1024) * 100 / count;
            const p = (c * 100 / count);
            if (p > pprev) {
                std.debug.print("{}% step {}\n", .{ p, c });
            }
        }
        c += 1;
    }
}

fn linearlySpreadHashes(cache: *[CacheSize]u8) [4]u16 {
    var max: u16 = 0;
    var count0: u16 = 0;
    var count1: u16 = 0;
    var currentRemaining: u16 = 0;
    for (cache) |v| {
        currentRemaining += v;
        if (currentRemaining > max) {
            @branchHint(.unlikely);
            count0 = 1;
            if (currentRemaining == max + 1) {
                count1 = count0;
            } else {
                count1 = 0;
            }
            max = currentRemaining;
        } else if (currentRemaining == max) {
            @branchHint(.unlikely);
            count0 += 1;
        } else if (currentRemaining == max - 1) {
            @branchHint(.unlikely);
            count1 += 1;
        }
        currentRemaining -|= 1;
    }
    return .{ max, count0, count1, 0 };
}

fn generateTokens(io: std.Io, input: []const u8, output: []const u8) !void {
    const file = try std.Io.Dir.cwd().openFile(io, input, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    if (stat.size == 0) return error.InvalidMerges;
    const data = try std.posix.mmap(null, stat.size, .{ .READ = true }, .{ .TYPE = .PRIVATE }, file.handle, 0);
    defer std.posix.munmap(data);
    const parsed = if (std.mem.startsWith(u8, data, "GGUF")) blk: {
        var it = try @import("gguf_merges.zig").Iterator.init(data);
        break :blk tokensFromIterator(&it) orelse return error.InvalidMerges;
    } else tokens(data) orelse return error.InvalidMerges;
    try std.Io.Dir.cwd().writeFile(io, .{
        .data = &serializeHash(&buildHashStructure(&parsed)),
        .sub_path = output,
    });
}

test "parsing" {
    const parsed = tokens("Ġ t\nĠ a\nĠt a\n").?;
    try std.testing.expectEqual(TokenValue{ .token1 = ' ', .token2 = 't', .resultToken = 256 }, parsed[0]);
    try std.testing.expectEqual(TokenValue{ .token1 = 256, .token2 = 'a', .resultToken = 258 }, parsed[2]);
    for (parsed[3..]) |token| try std.testing.expectEqual(TokenValue{}, token);
    try std.testing.expect(tokens("missing\n") == null);
    try std.testing.expect(tokens("unknown t\n") == null);
}

test "parsing all merge slots and rejecting overflow" {
    const RepeatedMerges = struct {
        remaining: usize,
        fn next(self: *@This()) ?[]const u8 {
            if (self.remaining == 0) return null;
            self.remaining -= 1;
            return "a b";
        }
    };
    var full = RepeatedMerges{ .remaining = Size };
    const parsed = tokensFromIterator(&full).?;
    try std.testing.expectEqual(@as(u16, Size + 255), parsed[Size - 1].resultToken);
    try std.testing.expectEqual(@as(usize, 0), full.remaining);
    var overflow = RepeatedMerges{ .remaining = Size + 1 };
    try std.testing.expect(tokensFromIterator(&overflow) == null);
}

test "parsing partial grammar hash ignores padding" {
    const parsed = tokens("a b\nb c\n").?;
    const h = buildHashStructure(&parsed);
    try std.testing.expectEqual(@as(?u16, 256), h.lookup('a', 'b'));
    try std.testing.expectEqual(@as(?u16, 257), h.lookup('b', 'c'));
    try std.testing.expectEqual(@as(?u16, null), h.lookup(0, 0));
    try std.testing.expectEqual(@as(?u16, null), h.lookup('x', 'y'));
}

pub fn main(init: std.process.Init) !void {
    var it = init.minimal.args.iterate();
    _ = it.skip();
    return run(init, it);
}

pub fn run(init: std.process.Init, args: anytype) !void {
    var it = args;
    const command = it.next();
    if (command) |arg| {
        if (std.mem.eql(u8, arg, "extract-merges")) {
            const input = it.next() orelse return error.MissingGgufPath;
            const output = it.next() orelse "merges.txt";
            if (it.next() != null) return error.TooManyArguments;
            try @import("gguf_merges.zig").extract(init.io, input, output);
            return;
        }
        if (std.mem.eql(u8, arg, "generate")) {
            const input = it.next() orelse "src/merges.txt";
            const output = it.next() orelse "tokens";
            if (it.next() != null) return error.TooManyArguments;
            try generateTokens(init.io, input, output);
            return;
        }
    }
    const is_tokenize = if (command) |arg| std.mem.eql(u8, arg, "tokenize") or std.mem.eql(u8, arg, "tokenize-only") else false;
    if (!is_tokenize) {
        const input = it.next() orelse "src/merges.txt";
        if (it.next() != null) return error.TooManyArguments;
        const source = try std.Io.Dir.cwd().readFileAlloc(init.io, input, init.arena.allocator(), .unlimited);
        const parsed = tokens(source) orelse return error.InvalidMerges;
        var grammar = parsed;
        try tryHashTok(init.io, &grammar, if (command) |arg| try std.fmt.parseInt(usize, arg, 10) else 10_000, 10);
        return;
    }
    const file = try std.Io.Dir.cwd().openFile(init.io, "tokens", .{});
    defer file.close(init.io);
    const stat = try file.stat(init.io);
    if (stat.size != TableBytes) return error.InvalidTokenTable;
    const mapped = try std.posix.mmap(null, TableBytes, .{ .READ = true }, .{ .TYPE = .PRIVATE }, file.handle, 0);
    defer std.posix.munmap(mapped);
    const h = try loadHash(mapped);
    if (command) |arg| {
        if (std.mem.eql(u8, arg, "tokenize")) {
            // parse stdin
            var buf: [4096]u8 = undefined;
            var r = std.Io.File.stdin().reader(init.io, &buf);
            const d = try r.interface.allocRemaining(init.arena.allocator(), .unlimited);
            var w = try TokenizeHeap.init(d, &h, init.arena.allocator());
            while (w.next()) {}
            var f = std.Io.File.stdout().writer(init.io, &buf);
            try f.interface.print("{f}", .{w});
            try f.interface.print("{any}\n", .{w.tokens});
            try f.flush();
            return;
        } else if (std.mem.eql(u8, arg, "tokenize-only")) {
            var buf: [4096]u8 = undefined;
            var r = std.Io.File.stdin().reader(init.io, &buf);
            const d = try r.interface.allocRemaining(init.arena.allocator(), .unlimited);
            var w = try TokenizeHeap.init(d, &h, init.arena.allocator());
            while (w.next()) {}
            var f = std.Io.File.stdout().writer(init.io, &buf);
            for (w.tokens) |u| {
                if (u < 61000) {
                    const tt =
                        if (u <= 32) u + 188 else if (u <= 126) u - 33 else if (u <= 160) u + 94 else if (u <= 172) u - 67 else if (u == 173) 255 else if (u <= 255) u - 68 else u;

                    try f.interface.print("{} ", .{tt});
                }
            }
            try f.flush();
            return;
        }
    }
}

const Bucket = extern struct {
    from: [8]u32 align(64),
    to: [8]u16,
    padding: [16]u8 = .{0} ** 16,

    fn add(this: *Bucket, index: u8, t: TokenValue) void {
        const p: *[8]u32 = @ptrCast(&this.from);
        p[index] = @as(u32, t.token1) | (@as(u32, t.token2) << 16);
        this.to[index] = t.resultToken;
    }
};

const Buckets = 6252;

const HashData = struct {
    hashAlgo: u32,
    buckets: [Buckets]Bucket,
    indices: [CacheSize]u16,
    oversizeStartIndex: u16,

    fn view(this: *const HashData) HashView {
        return .{ .hashAlgo = this.hashAlgo, .buckets = &this.buckets, .indices = &this.indices, .oversizeStartIndex = this.oversizeStartIndex };
    }

    fn lookup(this: *const HashData, tok1: u16, tok2: u16) ?u16 {
        return this.view().lookup(tok1, tok2);
    }
};

const HashView = struct {
    hashAlgo: u32,
    buckets: *const [Buckets]Bucket,
    indices: *const [CacheSize]u16,
    oversizeStartIndex: u16,

    fn view(this: *const HashView) HashView {
        return this.*;
    }

    fn lookup(this: *const HashView, tok1: u16, tok2: u16) ?u16 {
        const h = this.indices[hash(tok1, tok2, this.hashAlgo)];
        const bools = @as(@Vector(8, u32), this.buckets[h].from) == @as(@Vector(8, u32), @splat(@as(u32, tok1) | (@as(u32, tok2) << 16)));
        const index = @ctz(@as(u8, @bitCast(bools)));

        if (index == 8 and h >= this.oversizeStartIndex) {
            @branchHint(.unlikely);
            const bools2 = @as(@Vector(8, u32), this.buckets[h + 1].from) == @as(@Vector(8, u32), @splat(@as(u32, tok1) | (@as(u32, tok2) << 16)));
            const index2 = @ctz(@as(u8, @bitCast(bools2)));
            if (index2 == 8) return null;
            return this.buckets[h + 1].to[index2];
        } else {
            if (index == 8) return null;
            return this.buckets[h].to[index];
        }
    }
};

// Version 1: 64-byte header, little-endian indices, then 64-byte buckets.
const IndexOffset = 64;
const BucketOffset = IndexOffset + CacheSize * 2;
const TableBytes = BucketOffset + Buckets * 64;
const TableMagic = "TIKHASH\x00";

fn serializeHash(h: *const HashData) [TableBytes]u8 {
    var bytes: [TableBytes]u8 = @splat(0);
    @memcpy(bytes[0..8], TableMagic);
    const fields = [_]u32{ 1, h.hashAlgo, IndexOffset, CacheSize, BucketOffset, Buckets, h.oversizeStartIndex, TableBytes };
    for (fields, 0..) |v, i| std.mem.writeInt(u32, bytes[8 + i * 4 ..][0..4], v, .little);
    for (h.indices, 0..) |v, i| std.mem.writeInt(u16, bytes[IndexOffset + i * 2 ..][0..2], v, .little);
    for (h.buckets, 0..) |bucket, i| {
        const offset = BucketOffset + i * 64;
        for (bucket.from, 0..) |v, j| std.mem.writeInt(u32, bytes[offset + j * 4 ..][0..4], v, .little);
        for (bucket.to, 0..) |v, j| std.mem.writeInt(u16, bytes[offset + 32 + j * 2 ..][0..2], v, .little);
    }
    return bytes;
}

fn loadHash(bytes: []align(64) const u8) !HashView {
    if (@import("builtin").target.cpu.arch.endian() != .little) return error.UnsupportedTokenTableEndian;
    if (bytes.len != TableBytes or !std.mem.eql(u8, bytes[0..8], TableMagic)) return error.InvalidTokenTable;
    var fields: [8]u32 = undefined;
    for (&fields, 0..) |*v, i| v.* = std.mem.readInt(u32, bytes[8 + i * 4 ..][0..4], .little);
    if (fields[0] != 1) return error.UnsupportedTokenTableVersion;
    if (fields[2] != IndexOffset or fields[3] != CacheSize or fields[4] != BucketOffset or fields[5] != Buckets or fields[6] >= Buckets or fields[7] != TableBytes) return error.InvalidTokenTable;
    const indices: *const [CacheSize]u16 = @ptrCast(@alignCast(bytes.ptr + IndexOffset));
    for (indices) |index| {
        if (index >= Buckets or (index >= fields[6] and index + 1 >= Buckets)) return error.InvalidTokenTable;
    }
    comptime {
        std.debug.assert(@sizeOf(Bucket) == 64);
        std.debug.assert(@offsetOf(Bucket, "to") == 32);
    }
    return .{ .hashAlgo = fields[1], .indices = indices, .buckets = @ptrCast(@alignCast(bytes.ptr + BucketOffset)), .oversizeStartIndex = @intCast(fields[6]) };
}

test "parsing persisted hash round trip and validation" {
    const grammar = tokens("a b\nb c\nab c\n").?;
    const original = buildHashStructure(&grammar);
    var bytes align(64) = serializeHash(&original);
    const loaded = try loadHash(&bytes);
    for (grammar) |token| {
        if (token.resultToken == 0) continue;
        try std.testing.expectEqual(@as(?u16, token.resultToken), loaded.lookup(token.token1, token.token2));
    }
    try std.testing.expectEqual(@as(?u16, null), loaded.lookup('x', 'y'));
    var before = try TokenizeHeap.init("abc ab bc xyz", &original, std.testing.allocator);
    defer before.deinit(std.testing.allocator);
    var after = try TokenizeHeap.init("abc ab bc xyz", &loaded, std.testing.allocator);
    defer after.deinit(std.testing.allocator);
    while (before.next()) {}
    while (after.next()) {}
    try std.testing.expectEqualSlices(u16, before.tokens, after.tokens);
    const ids = try encodeHash(std.testing.allocator, &loaded, "abc ab bc xyz");
    defer std.testing.allocator.free(ids);
    try std.testing.expectEqualSlices(u16, &.{ 258, 220, 256, 220, 257, 220, 87, 88, 89 }, ids);
    try std.testing.expectError(error.InvalidTokenTable, loadHash(bytes[0..64]));
    bytes[0] = 0;
    try std.testing.expectError(error.InvalidTokenTable, loadHash(&bytes));
    bytes[0] = 'T';
    bytes[8] = 2;
    try std.testing.expectError(error.UnsupportedTokenTableVersion, loadHash(&bytes));
    bytes[8] = 1;
    std.mem.writeInt(u16, bytes[IndexOffset..][0..2], Buckets - 1, .little);
    try std.testing.expectError(error.InvalidTokenTable, loadHash(&bytes));
}

test "parsing persisted overflow bucket" {
    const grammar = tokens("a b\n").?;
    var original = buildHashStructure(&grammar);
    original.indices[hash('a', 'b', original.hashAlgo)] = original.oversizeStartIndex;
    original.buckets[original.oversizeStartIndex + 1].add(7, grammar[0]);
    var bytes align(64) = serializeHash(&original);
    const loaded = try loadHash(&bytes);
    try std.testing.expectEqual(@as(?u16, 256), loaded.lookup('a', 'b'));
    bytes[16] = 0;
    try std.testing.expectError(error.InvalidTokenTable, loadHash(&bytes));
}

fn buildHashStructure(toks: *const [Size]TokenValue) HashData {
    //const fixedMul = 0xACDD9B61;
    //const fixedMul = 0xE873DC86;
    //const fixedMul = 0x619BDDAC;
    const fixedMul = 0x3E5416A6; //  A616543E;
    var counts: [CacheSize]u8 = .{0} ** CacheSize;
    const b = loop1(toks, fixedMul, &counts, 11);
    if (!b) {
        std.debug.panic("hash counts too large", .{});
    }

    var bucketFilled: [Buckets]u8 = .{0} ** Buckets;
    var bucketActuallyFilled: [Buckets]u8 = .{0} ** Buckets;
    var completelyFilledUpTo: u16 = 0;
    var result: HashData = .{
        .buckets = .{Bucket{ .from = .{0xFFFFFFFF} ** 8, .to = .{0} ** 8 }} ** Buckets,
        .indices = .{0xFFFF} ** CacheSize,
        .oversizeStartIndex = 6222,
        .hashAlgo = fixedMul,
    };

    for (toks) |t| {
        if (t.resultToken == 0) continue;
        const h = hash(t.token1, t.token2, fixedMul);
        var index = result.indices[h];
        if (index == 0xFFFF) {
            const count = counts[h];
            if (count > 8) {
                index = result.oversizeStartIndex;
                while (bucketFilled[index] + bucketFilled[index + 1] + count > 16) {
                    index += 1;
                }
                result.indices[h] = index;
                const remaining = 8 - bucketFilled[index];
                bucketFilled[index] = 8;
                counts[h] = remaining;
                bucketFilled[index + 1] = count - remaining;
            } else {
                index = completelyFilledUpTo;
                while (bucketFilled[index] + count > 8) {
                    index += 1;
                }
                result.indices[h] = index;
                bucketFilled[index] += count;
                while (bucketFilled[completelyFilledUpTo] == 8) {
                    completelyFilledUpTo += 1;
                }
            }
        } else if (index == 0xFFFE) {
            continue;
        }
        if (counts[h] > 0) {
            counts[h] -= 1;
        } else {
            index += 1;
        }
        const bucketIndex = bucketActuallyFilled[index];
        bucketActuallyFilled[index] += 1;
        result.buckets[index].add(bucketIndex, t);
    }
    var partiallyFilledUpTo = completelyFilledUpTo;
    while (bucketFilled[partiallyFilledUpTo] > 0) {
        partiallyFilledUpTo += 1;
    }

    for (&result.indices) |*bv| {
        if (bv.* == 0xFFFF) {
            bv.* = 0;
        }
    }

    return result;
}

test "persisted full table lookups" {
    const file = try std.Io.Dir.cwd().openFile(std.testing.io, "tokens", .{});
    defer file.close(std.testing.io);
    const stat = try file.stat(std.testing.io);
    if (stat.size != TableBytes) return error.InvalidTokenTable;
    const mapped = try std.posix.mmap(null, TableBytes, .{ .READ = true }, .{ .TYPE = .PRIVATE }, file.handle, 0);
    defer std.posix.munmap(mapped);
    const h = try loadHash(mapped);
    const expected = try encodeHash(std.testing.allocator, &h, "abc ab bc xyz");
    defer std.testing.allocator.free(expected);
    const actual = try encodeTable(std.testing.io, std.testing.allocator, "tokens", "abc ab bc xyz");
    defer std.testing.allocator.free(actual);
    try std.testing.expectEqualSlices(u16, expected, actual);
    for (h.buckets) |bucket| {
        for (bucket.from, bucket.to) |pair, result| {
            if (pair == 0xFFFFFFFF) continue;
            try std.testing.expectEqual(@as(?u16, result), h.lookup(@truncate(pair), @truncate(pair >> 16)));
        }
    }
}

test "u4 bitcast ordering" {
    const bytes: [1]u8 = .{0xAB};
    const v: @Vector(2, u4) = @bitCast(bytes);

    try std.testing.expectEqual(@as(u4, 0xB), v[0]);
    try std.testing.expectEqual(@as(u4, 0xA), v[1]);
}

const HeapData = struct {
    resultId: u16,
    lengthFirstToken: u16,
    from: usize,
    to: usize,

    fn lessThan(a: HeapData, b: HeapData) bool {
        return a.resultId < b.resultId;
    }
};

test "heapdatasixe" {
    try std.testing.expectEqual(248, @sizeOf([10]HeapData));
}

const TokenHeap = struct {
    tags: []u16,
    data: []HeapData,
    size: usize,

    const B = 128;
    const V = @Vector(B, u16);

    fn init(data: []HeapData, tags: []u16, size: usize) TokenHeap {
        var result = TokenHeap{
            .data = data,
            .size = size,
            .tags = tags,
        };
        var i = data.len;
        while (i > 0) {
            i -= 1;
            tags[i] = data[i].resultId;
            result.bubbleDown(i);
        }
        return result;
    }

    fn peekIndex(this: *@This()) ?usize {
        if (this.size < B) {
            @branchHint(.unlikely);
            if (this.size == 0) return null;
            var min = this.tags[0];
            var index: usize = 0;
            for (1..this.size) |i| {
                if (this.tags[i] < min) {
                    index = i;
                    min = this.tags[i];
                }
            }
            return index;
        }
        const tags0: V = this.tags[0..B].*;
        const min = @reduce(.Min, tags0);
        const bits: @Int(.unsigned, B) = @bitCast(tags0 == @as(V, @splat(min)));
        return @ctz(bits);
    }

    fn peek(this: *@This()) ?*const HeapData {
        return &this.data[this.peekIndex() orelse return null];
    }

    fn bubbleDown(this: *@This(), indexX: usize) void {
        var index = indexX;
        while (true) {
            const a = B * (index + 1);
            if (a + B < this.size) {
                @branchHint(.likely);
                const v: V = this.tags[a..][0..B].*;
                const min = @reduce(.Min, v);

                if (this.tags[index] < min) {
                    const bits: @Int(.unsigned, B) = @bitCast(v == @as(V, @splat(min)));
                    const swapIndex = a + @ctz(bits);
                    const ele = this.data[index];
                    this.data[index] = this.data[swapIndex];
                    this.data[swapIndex] = ele;
                    this.tags[swapIndex] = this.tags[index];
                    this.tags[index] = min;
                    index = swapIndex;
                } else {
                    return;
                }
            } else {
                @branchHint(.unlikely);
                var min = this.tags[index];
                var j = a;
                var selected: usize = 0;
                while (j < this.size) {
                    if (this.tags[j] < min) {
                        min = this.tags[j];
                        selected = j;
                    }
                    j += 1;
                }
                if (selected != 0) {
                    const ele = this.data[index];
                    this.data[index] = this.data[selected];
                    this.data[selected] = ele;
                    this.tags[selected] = this.tags[index];
                    this.tags[index] = min;
                    index = selected;
                } else {
                    return;
                }
            }
        }
    }

    fn bubbleDownWithEle(this: *@This(), element: HeapData, indexX: usize) void {
        var index = indexX;
        var ele = element;
        while (true) {
            if (ele.resultId < this.data[index].resultId) {
                const extract = this.data[index];
                this.data[index] = ele;
                this.tags[index] = ele.resultId;
                ele = extract;
            }

            const a = B * (index + 1);

            if (a + B < this.size) {
                @branchHint(.likely);
                const v: V = this.tags[a..][0..B].*;
                const min = @reduce(.Min, v);

                if (this.tags[index] <= min) {
                    index = a;
                    continue;
                }

                const bits: @Int(.unsigned, B) = @bitCast(v == @as(V, @splat(min)));
                const swapIndex = a + @ctz(bits);

                const e = this.data[index];
                this.data[index] = this.data[swapIndex];
                this.data[swapIndex] = e;
                this.tags[swapIndex] = this.tags[index];
                this.tags[index] = min;
                index = swapIndex;
            } else {
                @branchHint(.unlikely);
                var min: u16 = 0xFFFF;
                var j = a;
                var swapIndex: usize = 0;
                while (j < this.size) {
                    if (this.tags[j] < min) {
                        min = this.tags[j];
                        swapIndex = j;
                    }
                    j += 1;
                }

                if (swapIndex > 0) {
                    const e = this.data[index];
                    this.data[index] = this.data[swapIndex];
                    this.data[swapIndex] = e;
                    this.tags[swapIndex] = this.tags[index];
                    this.tags[index] = min;
                    index = swapIndex;
                } else {
                    this.data[this.size] = ele;
                    this.tags[this.size] = ele.resultId;
                    this.size += 1;
                    return;
                }
            }
        }
    }

    fn drop(this: *@This()) void {
        const index = this.peekIndex() orelse std.debug.panic("PANIC", .{});
        this.size -= 1;
        this.data[index] = this.data[this.size];
        this.tags[index] = this.tags[this.size];
        @call(.never_inline, bubbleDown, .{ this, index });
    }

    fn replace1(this: *@This(), ele: HeapData) void {
        const index = this.peekIndex() orelse std.debug.panic("PANIC", .{});
        this.data[index] = ele;
        this.tags[index] = ele.resultId;
        this.bubbleDown(index);
    }

    fn replace(this: *@This(), newEle1: HeapData, newEle2: HeapData) void {
        const index = this.peekIndex() orelse std.debug.panic("PANIC", .{});
        this.data[index] = newEle1;
        this.tags[index] = newEle1.resultId;
        this.bubbleDownWithEle(newEle2, index);
    }
};

const TokenizeHeap = struct {
    original: []const u8,
    heap: TokenHeap,
    hd: HashView,
    tokens: []u16,

    fn next(this: *@This()) bool {
        if (this.heap.peek()) |e| {
            if (this.tokens[e.from] != 0xFFFF and (e.to == this.tokens.len or this.tokens[e.to] != 0xFFFF)) {
                this.tokens[e.from + e.lengthFirstToken] = 0xFFFF;
                var replacements: [2]HeapData = undefined;
                var replacementSize: usize = 0;
                if (e.to < this.tokens.len) {
                    if (this.hd.lookup(e.resultId, this.tokens[e.to])) |r| {
                        var newTo = e.to + 1;
                        while (newTo < this.tokens.len and this.tokens[newTo] == 0xFFFF) {
                            newTo += 1;
                        }

                        replacements[0] = HeapData{
                            .from = e.from,
                            .to = newTo,
                            .lengthFirstToken = @intCast(e.to - e.from),
                            .resultId = r,
                        };
                        replacementSize += 1;
                    }
                }

                if (e.from > 0) {
                    var previous = e.from - 1;
                    while (this.tokens[previous] == 0xFFFF) {
                        previous -= 1;
                    }
                    if (this.hd.lookup(this.tokens[previous], e.resultId)) |r| {
                        replacements[replacementSize] = HeapData{
                            .from = previous,
                            .to = e.to,
                            .lengthFirstToken = @intCast(e.from - previous),
                            .resultId = r,
                        };
                        replacementSize += 1;
                    }
                }

                this.tokens[e.from] = e.resultId;

                if (replacementSize == 0) {
                    this.heap.drop();
                } else if (replacementSize == 1) {
                    this.heap.replace1(replacements[0]);
                } else {
                    this.heap.replace(replacements[0], replacements[1]);
                }
            } else {
                this.heap.drop();
            }
            return true;
        } else {
            return false;
        }
    }

    pub fn format(this: *const @This(), writer: *std.Io.Writer) !void {
        for (0..this.original.len) |i| {
            if (this.tokens[i] != 0xFFFF) {
                try writer.writeByte('|');
            }
            try writer.writeByte(this.original[i]);
        }
        try writer.writeByte('|');
    }

    fn init(str: []const u8, hashData: anytype, alloc: std.mem.Allocator) !TokenizeHeap {
        const t = try alloc.alloc(u16, str.len);
        const heap = try alloc.alloc(HeapData, 2 * str.len);
        var heapSize: usize = 0;
        var i: usize = 0;
        if (str.len > 0) t[str.len - 1] = str[str.len - 1];
        while (i + 1 < str.len) {
            t[i] = str[i];
            const curr = str[i];
            const n = str[i + 1];
            if (hashData.lookup(curr, n)) |tt| {
                heap[heapSize] = .{
                    .from = i,
                    .to = i + 2,
                    .lengthFirstToken = 1,
                    .resultId = tt,
                };
                heapSize += 1;
            }
            i += 1;
        }
        return .{
            .original = str,
            .tokens = t,
            .heap = TokenHeap.init(heap, try alloc.alloc(u16, 2 * str.len), heapSize),
            .hd = hashData.view(),
        };
    }

    fn deinit(this: *@This(), alloc: std.mem.Allocator) void {
        alloc.free(this.heap.data);
        alloc.free(this.heap.tags);
        alloc.free(this.tokens);
    }
};

test "Sample tokenization" {
    const file = try std.Io.Dir.cwd().openFile(std.testing.io, "tokens", .{});
    defer file.close(std.testing.io);
    const stat = try file.stat(std.testing.io);
    if (stat.size != TableBytes) return error.InvalidTokenTable;
    const mapped = try std.posix.mmap(null, TableBytes, .{ .READ = true }, .{ .TYPE = .PRIVATE }, file.handle, 0);
    defer std.posix.munmap(mapped);
    const h = try loadHash(mapped);
    var th = try TokenizeHeap.init("Test string", &h, std.testing.allocator);
    defer th.deinit(std.testing.allocator);
    while (th.next()) {
        std.debug.print("{f}\n", .{th});
    }
    try std.testing.expectFmt("|Test| string|", "{f}", .{th});
}

pub fn encodeTableOrGenerate(io: std.Io, alloc: std.mem.Allocator, model: []const u8, path: []const u8, text: []const u8) ![]u16 {
    return encodeTable(io, alloc, path, text) catch |err| switch (err) {
        error.FileNotFound => blk: {
            try generateTokens(io, model, path);
            break :blk try encodeTable(io, alloc, path, text);
        },
        else => return err,
    };
}

test "parsing missing table generation and existing table reuse" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const alloc = std.testing.allocator;
    const source = try std.fmt.allocPrint(alloc, ".zig-cache/tmp/{s}/merges.txt", .{tmp.sub_path});
    defer alloc.free(source);
    const table = try std.fmt.allocPrint(alloc, ".zig-cache/tmp/{s}/tokens", .{tmp.sub_path});
    defer alloc.free(table);
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "merges.txt", .data = "a b\nb c\nab c\n" });
    const generated = try encodeTableOrGenerate(std.testing.io, alloc, source, table, "abc");
    defer alloc.free(generated);
    try std.testing.expectEqualSlices(u16, &.{258}, generated);
    // Reuse must succeed even when the original source is no longer available.
    try tmp.dir.deleteFile(std.testing.io, "merges.txt");
    const reused = try encodeTableOrGenerate(std.testing.io, alloc, source, table, "abc");
    defer alloc.free(reused);
    try std.testing.expectEqualSlices(u16, generated, reused);
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "tokens", .data = "invalid" });
    try std.testing.expectError(error.InvalidTokenTable, encodeTableOrGenerate(std.testing.io, alloc, source, table, "abc"));
}

// The mapping only needs to live until encoding finishes; returned IDs are owned by alloc.
pub fn encodeTable(io: std.Io, alloc: std.mem.Allocator, path: []const u8, text: []const u8) ![]u16 {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    if (stat.size != TableBytes) return error.InvalidTokenTable;
    const mapped = try std.posix.mmap(null, TableBytes, .{ .READ = true }, .{ .TYPE = .PRIVATE }, file.handle, 0);
    defer std.posix.munmap(mapped);
    const h = try loadHash(mapped);
    return encodeHash(alloc, &h, text);
}

// Uses the same merge engine and ID mapping as tokenize-only, without a tokens file.
// Allocations belong to the caller's arena.
pub fn encodeGguf(alloc: std.mem.Allocator, data: []const u8, text: []const u8) ![]u16 {
    var iterator = try @import("gguf_merges.zig").Iterator.init(data);
    const grammar = try alloc.create([Size]TokenValue);
    grammar.* = tokensFromIterator(&iterator) orelse return error.InvalidMerges;
    const h = buildHashStructure(grammar);
    return encodeHash(alloc, &h.view(), text);
}

fn encodeHash(alloc: std.mem.Allocator, h: *const HashView, text: []const u8) ![]u16 {
    var heap = try TokenizeHeap.init(text, h, alloc);
    defer heap.deinit(alloc);
    while (heap.next()) {}
    var result: std.ArrayList(u16) = .empty;
    errdefer result.deinit(alloc);
    for (heap.tokens) |u| {
        if (u == 0xFFFF) continue;
        const id = if (u <= 32) u + 188 else if (u <= 126) u - 33 else if (u <= 160) u + 94 else if (u <= 172) u - 67 else if (u == 173) 255 else if (u <= 255) u - 68 else u;
        try result.append(alloc, id);
    }
    return result.toOwnedSlice(alloc);
}

test "parsing empty and single-byte prompts" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const h: HashData = undefined; // Neither input has a pair to look up.
    var empty = try TokenizeHeap.init("", &h, arena.allocator());
    try std.testing.expect(!empty.next());
    try std.testing.expectEqual(@as(usize, 0), empty.tokens.len);
    var single = try TokenizeHeap.init("a", &h, arena.allocator());
    try std.testing.expect(!single.next());
    try std.testing.expectEqualSlices(u16, &.{'a'}, single.tokens);
}
