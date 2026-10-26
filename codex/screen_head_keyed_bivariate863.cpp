// Reuse the independently tested module elimination without changing the
// earlier input-XOR accelerator or invalidating its source fingerprint.
#define main input_xor_accelerator_main
#include "screen_head_bivariate863.cpp"
#undef main

std::vector<Row> keyed_matrix(int a, int b, int key, const std::vector<int>& samples,
                             bool positive) {
    std::vector<Row> matrix;
    for (int x: samples) {
        int q = ((a*x+b) & mask) ^ key;
        std::array<int, 5> xp{1}, qp{1};
        for (int i = 1; i <= 4; ++i) {
            xp[i] = xp[i-1]*x & mask;
            qp[i] = qp[i-1]*q & mask;
        }
        Row row;
        for (int i = 0; i <= 4; ++i)
            for (int j = 0; j <= 4-i; ++j) row.push_back(xp[i]*qp[j] & mask);
        int cq = ((6*x+19) & mask) ^ 42;
        int target = positive ? (17*x*cq+23*cq*cq+31*x+7) & mask : middle(x);
        row.push_back(target);
        for (int bit = 1; bit < 10; ++bit)
            row.push_back(((1 << bit)-2*(target & (1 << bit))) & mask);
        matrix.push_back(row);
    }
    return matrix;
}

int main(int argc, char** argv) {
    bool positive = argc > 1 && std::string(argv[1]) == "control";
    int count;
    if (!(std::cin >> count) || count < 1) return 2;
    std::vector<int> samples(count);
    for (int& x: samples) std::cin >> x;
    int a, b, key, tested = 0, survivors = 0;
    while (std::cin >> a >> b >> key) {
        auto keys = consistent_keys(keyed_matrix(a,b,key,samples,positive),15);
        ++tested;
        if (!keys.empty()) {
            ++survivors;
            std::cout << a << ' ' << b << ' ' << key << ' ' << keys.size();
            for (int output_key: keys) std::cout << ' ' << output_key;
            std::cout << '\n';
        }
        if (tested % 10000 == 0)
            std::cerr << "tested=" << tested << " survivors=" << survivors << '\n';
    }
    std::cerr << "completed tested=" << tested << " survivors=" << survivors << '\n';
    return 0;
}
