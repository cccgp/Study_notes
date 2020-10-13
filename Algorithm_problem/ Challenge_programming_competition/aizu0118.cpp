#include <iostream>
#include <vector>

using namespace std;

void dfs(vector<vector<char>> &s, vector<vector<int>> &v, char c, int x, int y, int W, int H) {
    if (!(x >= 0 && x < W && y >= 0 && y < H)) return;
    if (v[x][y] == 1 || s[x][y] != c) return;
    v[x][y] = 1;
    dfs(s, v, c, x - 1, y, W, H);
    dfs(s, v, c, x + 1, y, W, H);
    dfs(s, v, c, x, y - 1, W, H);
    dfs(s, v, c, x, y + 1, W, H);
}

int main() {
    int W, H;
    while ((cin >> W >> H) && W && H) {
        vector<vector<char>> s(W, vector<char>(H));
        for (int i = 0; i < W; i++) {
            for (int j = 0; j < H; j++) {
                cin >> s[i][j];
            }
        }
        vector<vector<int>> v(W, vector<int>(H, 0));
        int count = 0;
        for (int i = 0; i < W; i++) {
            for (int j = 0; j < H; j++) {
                if (v[i][j] == 0) {
                    dfs(s, v, s[i][j], i, j, W, H);
                    count++;
                }
            }
        }
        cout << count << endl;
    }

    return 0;
}
/*
10 10
####*****@
@#@@@@#*#*
@##***@@@*
#****#*@**
##@*#@@*##
*@@@@*@@@#
***#@*@##*
*@@@*@@##@
*@*#*@##**
@****#@@#@
0 0

 */