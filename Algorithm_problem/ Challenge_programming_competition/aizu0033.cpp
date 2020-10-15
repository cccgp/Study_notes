#include <iostream>
#include <vector>
#include <stack>
using namespace std;

int main() {
    int n;
    cin >> n;
    while(n--){
        vector<int> nums(10);
        stack<int> s1,s2;
        bool res = true;
        for(int i = 0;i < 10;i++) {
            cin >> nums[i];
            if(s1.empty() || nums[i] > s1.top()){
                s1.push(nums[i]);
            }else if(s2.empty() || nums[i] > s2.top()){
                s2.push(nums[i]);
            }else{
                res = false;
            }
        }
        if(res) cout << "YES" << endl;
        else    cout << "NO" << endl;
    }

    return 0;
}
/*
2
3 1 4 2 5 6 7 8 9 10
10 9 8 7 6 5 4 3 2 1

 */
