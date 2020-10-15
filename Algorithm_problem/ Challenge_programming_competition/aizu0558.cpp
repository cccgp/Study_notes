#include <iostream>
#include <vector>
#include <stack>
#include <queue>

using namespace std;

#define MS 1000 + 5


int main() {
    int H, W, N;
    cin >> H >> W >> N;
    char nums[MS][MS];

    int startX = 0;
    int startY = 0;
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            cin >> nums[i][j];
            if (nums[i][j] == 'S') {
                startX = i;
                startY = j;
                nums[i][j] = '.';
            }
        }
    }
    int res = 0;
    int dirs[4][2] = {{-1, 0},
                      {1,  0},
                      {0,  1},
                      {0,  -1}};
    queue<pair<pair<int, int>, int>> q;
    for (int k = 1; k <= N; k++) {
        vector<vector<int>> v(MS, vector<int>(MS, 0));
        q.push({{startX, startY}, 0});
        while (!q.empty()) {
            int x = q.front().first.first;
            int y = q.front().first.second;
            int z = q.front().second;
            q.pop();

            if (v[x][y] == 1) continue;
            v[x][y] = 1;

            if(nums[x][y] == k + '0'){
                startX = x;
                startY = y;
                res += z;
                break;
            }

            for (int j = 0; j < 4; j++) {
                int newX = x + dirs[j][0];
                int newY = y + dirs[j][1];
                if (newX >= 0 && newX < H && newY >= 0 && newY < W && (nums[newX][newY] != 'X')) {
                    q.push({{newX, newY}, z + 1});
                }
            }
        }
        while (!q.empty()) q.pop();
    }
    cout << res << endl;
    return 0;
}
/*
3 3 1
S..
...
..1

 */