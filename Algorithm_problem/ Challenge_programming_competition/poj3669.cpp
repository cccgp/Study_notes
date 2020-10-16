#include <iostream>
#include <vector>
#include <stack>
#include <queue>
#include <map>
#include <set>

using namespace std;

#define MS 300 + 5

int main() {

    int M;
    scanf("%d", &M);
    int x, y, t;
    vector<vector<int>> nums(MS, vector<int>(MS, 1001));

    int dirs[5][2] = {{-1, 0},
                      {1,  0},
                      {0,  1},
                      {0,  -1},
                      {0,  0}};
    for (int i = 0; i < M; i++) {
        scanf("%d %d %d", &x, &y, &t);
        nums[x][y] = min(t, nums[x][y]);
        for (int j = 0; j < 4; j++) {
            int newX = x + dirs[j][0];
            int newY = y + dirs[j][1];
            if (newX >= 0 && newY >= 0) {
                nums[newX][newY] = min(t, nums[newX][newY]);
            }
        }
    }
    queue<pair<pair<int, int>, int>> q;
    q.push(make_pair(make_pair(0, 0), 0));
    int res = -1;
    while (!q.empty()) {
        x = q.front().first.first;
        y = q.front().first.second;
        t = q.front().second;
        q.pop();

        if (nums[x][y] == 1001) {
            res = t;
            break;
        }
        // 这段代码值得学习
        for (int j = 0; j < 5; j++) {
            int newX = x + dirs[j][0];
            int newY = y + dirs[j][1];
            if (newX >= 0 && newY >= 0 && t + 1 < nums[newX][newY]) {
                if (nums[newX][newY] != 1001) nums[newX][newY] = t + 1;
                q.push(make_pair(make_pair(newX, newY), t + 1));
            }
        }
    }
    printf("%d\n", res);
    return 0;
}
/*
4
0 0 2
2 1 2
1 1 2
0 3 5

 */