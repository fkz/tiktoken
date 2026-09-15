const std = @import("std");

fn string(reader: *std.Io.Reader) ![]const u8 {
    return reader.take(try reader.takeInt(u64, .little));
}

fn skipValue(reader: *std.Io.Reader, kind: u32) error{ EndOfStream, ReadFailed, InvalidMetadata }!void {
    const size: usize = switch (kind) {
        0, 1, 7 => 1,
        2, 3 => 2,
        4, 5, 6 => 4,
        10, 11, 12 => 8,
        8 => {
            _ = try string(reader);
            return;
        },
        9 => {
            const element = try reader.takeInt(u32, .little);
            if (element == 9 or element > 12) return error.InvalidMetadata;
            const count = try reader.takeInt(u64, .little);
            for (0..count) |_| try skipValue(reader, element);
            return;
        },
        else => return error.InvalidMetadata,
    };
    _ = try reader.take(size);
}

// Returns the encoded string array, preserving merge priority and spelling.
fn findMerges(data: []const u8) ![]const u8 {
    var reader = std.Io.Reader.fixed(data);
    if (!std.mem.eql(u8, try reader.take(4), "GGUF")) return error.InvalidGguf;
    const version = try reader.takeInt(u32, .little);
    if (version != 2 and version != 3) return error.UnsupportedGgufVersion;
    _ = try reader.takeInt(u64, .little); // tensor count
    const count = try reader.takeInt(u64, .little);
    for (0..count) |_| {
        const key = try string(&reader);
        const kind = try reader.takeInt(u32, .little);
        if (std.mem.eql(u8, key, "tokenizer.ggml.merges")) {
            if (kind != 9 or try reader.takeInt(u32, .little) != 8) return error.InvalidMetadata;
            const length = try reader.takeInt(u64, .little);
            const start = reader.seek;
            for (0..length) |_| {
                const merge = try string(&reader);
                if (std.mem.indexOfAny(u8, merge, "\r\n") != null) return error.MergeContainsNewline;
            }
            return data[start..reader.seek];
        }
        try skipValue(&reader, kind);
    }
    return error.MergesNotFound;
}

pub const Iterator = struct {
    reader: std.Io.Reader,

    pub fn init(data: []const u8) !Iterator {
        return .{ .reader = std.Io.Reader.fixed(try findMerges(data)) };
    }

    pub fn next(self: *Iterator) ?[]const u8 {
        if (self.reader.seek == self.reader.buffer.len) return null;
        // findMerges validated all lengths before constructing this iterator.
        return string(&self.reader) catch unreachable;
    }
};

pub fn extract(io: std.Io, input: []const u8, output: []const u8) !void {
    const file = try std.Io.Dir.cwd().openFile(io, input, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    if (stat.size == 0) return error.InvalidGguf;
    const data = try std.posix.mmap(null, stat.size, .{ .READ = true }, .{ .TYPE = .PRIVATE }, file.handle, 0);
    defer std.posix.munmap(data);
    const merges = try findMerges(data);
    const out = try std.Io.Dir.cwd().createFile(io, output, .{ .exclusive = true });
    defer out.close(io);
    var buffer: [4096]u8 = undefined;
    var writer = out.writer(io, &buffer);
    var reader = std.Io.Reader.fixed(merges);
    while (reader.seek < merges.len) {
        try writer.interface.writeAll(try string(&reader));
        try writer.interface.writeByte('\n');
    }
    try writer.flush();
}

test "extract merges and reject missing or truncated metadata" {
    const header = "GGUF\x03\x00\x00\x00" ++ "\x00" ** 8;
    const metadata = "\x15\x00\x00\x00\x00\x00\x00\x00tokenizer.ggml.merges" ++
        "\x09\x00\x00\x00\x08\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00";
    const entries = "\x03\x00\x00\x00\x00\x00\x00\x00h e" ++
        "\x04\x00\x00\x00\x00\x00\x00\x00he l";
    const fixture = header ++ "\x01\x00\x00\x00\x00\x00\x00\x00" ++ metadata ++ entries;
    try std.testing.expectEqualStrings(entries, try findMerges(fixture));
    var it = try Iterator.init(fixture);
    try std.testing.expectEqualStrings("h e", it.next().?);
    try std.testing.expectEqualStrings("he l", it.next().?);
    try std.testing.expect(it.next() == null);
    try std.testing.expectError(error.EndOfStream, findMerges(fixture[0 .. fixture.len - 1]));
    try std.testing.expectError(error.MergesNotFound, findMerges(header ++ "\x00" ** 8));
}
