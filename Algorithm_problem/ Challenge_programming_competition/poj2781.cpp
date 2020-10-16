#include <iostream>
#include <vector>
#include <stack>
#include <algorithm>
#include <map>
#include <set>
#include <sstream>
#include <string>

using namespace std;

int main() {
    int n;
    cin >> n;
    string s;
    getline(cin, s);
    while (n--) {
        getline(cin, s);
        istringstream str(s);
        vector<int> nums;
        string out;
        while (str >> out) {
            nums.push_back(out[0] - '0');
        }

        int l = nums.size();
        int res = INT_MAX;
        do {
            int left = 0, right = 0;
            for (int i = 0; i < l / 2; i++) {
                left = left * 10 + nums[i];
            }
            for (int i = l / 2; i < l; i++) {
                right = right * 10 + nums[i];
            }
            // 仔细审题
            if ((nums[0] == 0 && left != 0) || (nums[l / 2] == 0 && right != 0)) {
                continue;
            }
            res = min(res, abs(left - right));

        } while (next_permutation(nums.begin(), nums.end()));
        cout << res << endl;
    }
    return 0;
}
/*
3
0 1 2 4 6 7
0 1
1 0 0
 */