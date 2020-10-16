#include <iostream>
#include <vector>
#include <stack>
#include <algorithm>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <queue>
#include <set>
#include <cstdlib>

using namespace std;

int solve(vector<char> &nums, int i, int j, int n) {
    int l = i;
    int r = j;
    while (true) {
        if (i + 1 <= j - 1) {
            if (nums[i + 1] < nums[j - 1]) {
                return l;
            } else if (nums[i + 1] > nums[j - 1]) {
                return r;
            } else {
                i++;
                j--;
            }
        } else {
            break;
        }
    }
    return l;
}

int main() {
    int n;
    cin >> n;
    vector<char> nums(n);
    for (int i = 0; i < n; i++) cin >> nums[i];

    string res;
    int i = 0, j = n - 1;
    while (i <= j) {
        if (nums[i] < nums[j]) {
            res += nums[i++];
        } else if (nums[i] > nums[j]) {
            res += nums[j--];
        } else {
            int k = solve(nums,i,j,n);
            res += nums[k];
            if(k == i) i++;
            else j--;
        }
    }
    int l = res.size();
    int left = 0;

    while(l > 80){
        cout << res.substr(left,80) << endl;
        left += 80;
        l -= 80;
    }
    cout << res.substr(left,80) << endl;

    return 0;
}
/*
6
A
C
D
B
C
B

 */