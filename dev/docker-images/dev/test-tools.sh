#!/usr/bin/env bash
set -euo pipefail

echo "🧪 Testing SQLite3 and Zig installation in devcontainer..."
echo "========================================================="

# Test SQLite3 command availability
echo ""
echo "📊 Testing SQLite3..."
if command -v sqlite3 >/dev/null 2>&1; then
    echo "✅ SQLite3 command found: $(which sqlite3)"
    echo "✅ SQLite3 version: $(sqlite3 --version)"
else
    echo "❌ SQLite3 command not found"
    exit 1
fi

# Test SQLite3 library availability
if pkg-config --exists sqlite3; then
    echo "✅ SQLite3 development libraries found"
    echo "   - Version: $(pkg-config --modversion sqlite3)"
    echo "   - Cflags: $(pkg-config --cflags sqlite3)"
    echo "   - Libs: $(pkg-config --libs sqlite3)"
else
    echo "❌ SQLite3 development libraries not found"
    exit 1
fi

# Test basic SQLite3 functionality
echo ""
echo "🧪 Testing basic SQLite3 functionality..."

# Create a temporary database
TEMP_DB="/tmp/test_sqlite3.db"
rm -f "$TEMP_DB"

# Test database creation and basic operations
sqlite3 "$TEMP_DB" <<EOF
CREATE TABLE IF NOT EXISTS test_table (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    value INTEGER
);

INSERT INTO test_table (name, value) VALUES
    ('test1', 42),
    ('test2', 84),
    ('test3', 126);

SELECT COUNT(*) as row_count FROM test_table;
SELECT * FROM test_table WHERE value > 50;
EOF

if [[ -f "$TEMP_DB" ]]; then
    echo "✅ Database file created successfully"

    # Verify data was inserted
    ROW_COUNT=$(sqlite3 "$TEMP_DB" "SELECT COUNT(*) FROM test_table;")
    if [[ "$ROW_COUNT" == "3" ]]; then
        echo "✅ Data insertion and query successful (found $ROW_COUNT rows)"
    else
        echo "❌ Data verification failed (expected 3 rows, found $ROW_COUNT)"
        exit 1
    fi

    # Clean up
    rm -f "$TEMP_DB"
    echo "✅ Cleanup completed"
else
    echo "❌ Database file was not created"
    exit 1
fi

# Test Zig command availability
echo ""
echo "⚡ Testing Zig..."
if command -v zig >/dev/null 2>&1; then
    echo "✅ Zig command found: $(which zig)"
    echo "✅ Zig version: $(zig version)"
else
    echo "❌ Zig command not found"
    exit 1
fi

# Test basic Zig functionality
echo ""
echo "🧪 Testing basic Zig functionality..."

# Create a simple Zig program
ZIG_FILE="/tmp/test_zig.zig"
ZIG_BINARY="/tmp/test_zig"

cat > "$ZIG_FILE" <<EOF
const std = @import("std");

pub fn main() !void {
    std.debug.print("Hello from Zig in Atlas devcontainer!\n", .{});
}
EOF

# Compile and run the Zig program
if zig build-exe "$ZIG_FILE" -femit-bin="$ZIG_BINARY"; then
    echo "✅ Zig compilation successful"

    if [[ -f "$ZIG_BINARY" ]]; then
        echo "✅ Zig binary created successfully"

        # Run the binary
        OUTPUT=$("$ZIG_BINARY" 2>&1)
        if [[ "$OUTPUT" == "Hello from Zig in Atlas devcontainer!" ]]; then
            echo "✅ Zig program execution successful: $OUTPUT"
        else
            echo "❌ Zig program output unexpected: $OUTPUT"
            exit 1
        fi

        # Clean up
        rm -f "$ZIG_FILE" "$ZIG_BINARY"
        echo "✅ Cleanup completed"
    else
        echo "❌ Zig binary was not created"
        exit 1
    fi
else
    echo "❌ Zig compilation failed"
    exit 1
fi

echo ""
echo "🎉 SQLite3 and Zig installation test completed successfully!"
echo "   Both SQLite3 and Zig are ready for Atlas ISA integration testing."
