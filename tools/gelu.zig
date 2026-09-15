const std = @import("std");
const V = @Vector(16, f32);

/// SIMD tanh approximation for GELU; not a correctly rounded libm replacement.
/// Truncate Lambert's continued fraction x/(1+x²/(3+...+x²/23)),
/// then evaluate its numerator and denominator in x² using Horner's method.
/// Above |x|=8, rounding to +/-1 introduces less than 2.3e-7 error.
pub fn tanh(x: V) V {
    const saturated = @abs(x) > @as(V, @splat(8.0));
    const a = @select(f32, saturated, @as(V, @splat(0.0)), x);
    const z = a * a;
    var p: V = @splat(78.0);
    inline for (.{ 75075.0, 18378360.0, 1571349780.0, 45831035250.0, 316234143225.0 }) |c| {
        p = @mulAdd(V, p, z, @as(V, @splat(c)));
    }
    var q: V = @splat(1.0);
    inline for (.{ 3003.0, 1351350.0, 192972780.0, 9820936125.0, 151242416325.0, 316234143225.0 }) |c| {
        q = @mulAdd(V, q, z, @as(V, @splat(c)));
    }
    const sign = @select(f32, x < @as(V, @splat(0.0)), @as(V, @splat(-1.0)), @as(V, @splat(1.0)));
    return @select(f32, saturated, sign, a * (p / q));
}

pub fn gelu(c: V) V {
    const one: V = @splat(1.0);
    const inner = (@as(V, @splat(@sqrt(2.0 / std.math.pi))) * c) *
        @mulAdd(V, @as(V, @splat(0.044715)), c * c, one);
    return (@as(V, @splat(0.5)) * c) * (one + tanh(inner));
}

test "vector tanh and GELU match scalar references across mixed lanes" {
    // Include the central region, tanh saturation boundary, and GELU tails.
    for (0..12501) |batch| {
        var values: [16]f32 = undefined;
        for (&values, 0..) |*value, lane| {
            value.* = (@as(f32, @floatFromInt(batch * 16 + lane)) - 100000.0) / 5000.0;
        }
        const ts: [16]f32 = tanh(values);
        const gs: [16]f32 = gelu(values);
        for (values, ts, gs) |x, t, g| {
            try std.testing.expectApproxEqAbs(std.math.tanh(x), t, 5e-7);
            const u: f32 = @sqrt(2.0 / std.math.pi);
            const reference = x / 2.0 * (1.0 + std.math.tanh(u * (x + 0.044715 * x * x * x)));
            try std.testing.expectApproxEqAbs(reference, g, 2e-6);
        }
    }
}

test "vector tanh special values" {
    const values: V = .{ 0.0, -0.0, std.math.inf(f32), -std.math.inf(f32), std.math.nan(f32), 1e30, -1e30, 8.0, -8.0, 8.001, -8.001, 1e-30, -1e-30, 1.0, -1.0, 0.5 };
    const result: [16]f32 = tanh(values);
    try std.testing.expectEqual(@as(u32, 0), @as(u32, @bitCast(result[0])));
    try std.testing.expectEqual(@as(u32, 0x80000000), @as(u32, @bitCast(result[1])));
    try std.testing.expectEqual(@as(f32, 1.0), result[2]);
    try std.testing.expectEqual(@as(f32, -1.0), result[3]);
    try std.testing.expect(std.math.isNan(result[4]));
    inline for (5..16) |i| {
        try std.testing.expectApproxEqAbs(std.math.tanh(values[i]), result[i], 5e-7);
    }
}
