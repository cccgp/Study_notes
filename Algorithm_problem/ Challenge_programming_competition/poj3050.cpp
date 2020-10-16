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

int dirs[4][2] = {{-1,0},{1,0},{0,-1},{0,1}};
set<int> st;
void bfs(int nums[5][5],int x,int y,int num,int count){
    if(count == 5){
        st.insert(num);
        return;
    }
    for(int i = 0;i < 4;i++){
        int nx = x+dirs[i][0];
        int ny = y+dirs[i][1];
        if(nx >= 0 && nx < 5 && ny >= 0 && ny < 5)
            bfs(nums,nx,ny,num*10 + nums[nx][ny],count+1);
    }
}
int main() {
    int nums[5][5];
    for(int i = 0;i < 5;i++){
        for(int j = 0;j < 5;j++){
            cin >> nums[i][j];
        }
    }
    for(int i = 0;i < 5;i++){
        for(int j = 0;j < 5;j++){
            bfs(nums,i,j,nums[i][j],0);
        }
    }
    cout << st.size() << endl;

    return 0;
}
/*
1 1 1 1 1
1 1 1 1 1
1 1 1 1 1
1 1 1 2 1
1 1 1 1 1

 */