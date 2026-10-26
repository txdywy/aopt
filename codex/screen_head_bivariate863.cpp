// Exact linear-module membership over Z/(2^11), with all normalized output
// XOR keys tested. Accelerator for the Python proof/replay driver.
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

constexpr int width = 11, mask = 2047, rhs_columns = 10;
using Row = std::vector<int>;

int middle(int x) {
    uint32_t y = uint32_t(x) ^ 0xc761c23cU;
    return (9U*((33U*y+0xe9f8cc1dU) ^ (16896U*y+0xaccf6200U)) + 0xfd7046c5U) & mask;
}

int inverse_odd(int v) {
    int result = 1;
    for (int i = 0; i < width; ++i) result = int(int64_t(result)*(2-int64_t(v)*result) & mask);
    return result;
}

void filter_keys(const Row& row, int columns, int divisor, std::vector<int>& keys) {
    if (divisor == 1 || keys.empty()) return;
    int m = divisor-1;
    std::array<int, 512> values{};
    values[0] = row[columns] & m;
    for (int n = 1; n < 512; ++n)
        values[n] = (values[n & (n-1)] + row[columns+1+__builtin_ctz(unsigned(n))]) & m;
    keys.erase(std::remove_if(keys.begin(), keys.end(),
        [&](int key) { return values[key >> 1] != 0; }), keys.end());
}

std::vector<int> consistent_keys(std::vector<Row> matrix, int columns) {
    std::vector<int> keys(512);
    for (int i = 0; i < 512; ++i) keys[i] = 2*i;
    int row = 0;
    for (int col = 0; col < columns && row < int(matrix.size()); ++col) {
        int factor = 4096, best_i = -1, best_j = -1;
        for (int i = row; i < int(matrix.size()); ++i) {
            for (int j = col; j < columns; ++j) {
                int value = matrix[i][j], low = value & -value;
                if (value && low < factor) { factor = low; best_i = i; best_j = j; }
            }
        }
        if (best_i == -1) break;
        for (int i = row; i < int(matrix.size()); ++i) {
            filter_keys(matrix[i], columns, factor, keys);
            if (keys.empty()) return keys;
        }
        std::swap(matrix[row], matrix[best_i]);
        for (auto& line: matrix) std::swap(line[col], line[best_j]);
        int inverse = inverse_odd(matrix[row][col]/factor);
        for (int j = col; j < columns+rhs_columns; ++j)
            matrix[row][j] = (matrix[row][j]*inverse) & mask;
        for (int i = row+1; i < int(matrix.size()); ++i) {
            int multiple = matrix[i][col]/factor;
            for (int j = col; j < columns+rhs_columns; ++j)
                matrix[i][j] = (matrix[i][j]-multiple*matrix[row][j]) & mask;
        }
        ++row;
    }
    for (int i = row; i < int(matrix.size()); ++i) {
        filter_keys(matrix[i], columns, 2048, keys);
        if (keys.empty()) break;
    }
    return keys;
}

std::vector<Row> build_matrix(int a, int b, int degree, const std::vector<int>& samples,
                             bool positive) {
    std::vector<Row> matrix;
    for (int x: samples) {
        int q = ((a*x+b) & mask) ^ x;
        std::vector<int> xp(degree+1, 1), qp(degree+1, 1);
        for (int i = 1; i <= degree; ++i) {
            xp[i] = xp[i-1]*x & mask;
            qp[i] = qp[i-1]*q & mask;
        }
        Row row;
        for (int i = 0; i <= degree; ++i)
            for (int j = 0; j <= degree-i; ++j) row.push_back(xp[i]*qp[j] & mask);
        // A fixed genuine quadratic positive-control target, independent of
        // tested a,b. Pair (6,19) must survive with output key zero.
        int cq = ((6*x+19) & mask) ^ x;
        int target = positive ? (17*x*cq+23*cq*cq+31*x+7) & mask : middle(x);
        row.push_back(target);
        for (int bit = 1; bit < 10; ++bit)
            row.push_back(((1 << bit)-2*(target & (1 << bit))) & mask);
        matrix.push_back(row);
    }
    return matrix;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "matrix") {
        int rows, columns;
        if (!(std::cin >> rows >> columns) || rows < 1 || columns < 1) return 2;
        std::vector<Row> matrix(rows, Row(columns+rhs_columns));
        for (auto& row: matrix) {
            for (int j = 0; j <= columns; ++j) std::cin >> row[j];
            int target = row[columns];
            for (int bit = 1; bit < 10; ++bit)
                row[columns+bit] = ((1 << bit)-2*(target & (1 << bit))) & mask;
        }
        auto keys = consistent_keys(matrix, columns);
        for (int key: keys) std::cout << key << ' ';
        std::cout << '\n';
        return 0;
    }
    int degree = argc > 1 ? std::stoi(argv[1]) : 4;
    bool positive = argc > 2 && std::string(argv[2]) == "control";
    if (degree < 1 || degree > 4) return 2;
    int count;
    if (!(std::cin >> count) || count < 1) return 2;
    std::vector<int> samples(count);
    for (int& x: samples) std::cin >> x;
    int a, b, tested = 0, survivors = 0;
    int columns = (degree+1)*(degree+2)/2;
    while (std::cin >> a >> b) {
        auto keys = consistent_keys(build_matrix(a,b,degree,samples,positive),columns);
        ++tested;
        if (!keys.empty()) {
            ++survivors;
            std::cout << a << ' ' << b << ' ' << keys.size();
            for (int key: keys) std::cout << ' ' << key;
            std::cout << '\n';
        }
        if (tested % 5000 == 0)
            std::cerr << "tested=" << tested << " survivors=" << survivors << '\n';
    }
    std::cerr << "completed tested=" << tested << " survivors=" << survivors << '\n';
}
