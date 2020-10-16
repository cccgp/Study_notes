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
    int m,n;
    cin >> m >> n;
    vector<int> nums(m);
    for(int i = 0;i < m;i++){
        nums[i] = i+1;
    }

    do{
        vector<int> newNums(nums);
        for(int i = 0;i < m-1;i++){
            for(int j = 0;j < m-i;j++){
                newNums[j] = newNums[j]+newNums[j+1];
            }
        }
        if(newNums[0] == n){
            for(int i = 0;i < m;i++){
                cout << nums[i] << " ";
            }
            cout << endl;
            break;
        }
    }while(next_permutation(nums.begin(),nums.end()));

    return 0;
}
/*
4 16
 */