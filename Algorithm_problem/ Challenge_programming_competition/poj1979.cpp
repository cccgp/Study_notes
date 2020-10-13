#include <iostream>
#include <vector>

using namespace std;

int dfs(vector<vector<char>> &s, int x, int y, int W, int H) {
    if (!(x >= 0 && x < W && y >= 0 && y < H)) return 0;
    if (s[x][y] == '#') return 0;
    s[x][y] = '#';
    int sum = dfs(s, x - 1, y, W, H) + dfs(s, x + 1, y, W, H) +
              dfs(s, x, y - 1, W, H) + dfs(s, x, y + 1, W, H);
    return sum + 1;
}

int main() {
    int W, H;
    while ((cin >> H >> W) && W && H) {
        vector<vector<char>> s(W, vector<char>(H));
        int x = 0, y = 0;
        for (int i = 0; i < W; i++) {
            for (int j = 0; j < H; j++) {
                cin >> s[i][j];
                if (s[i][j] == '@') {
                    x = i;
                    y = j;
                }
            }
        }
//        cout << x << " " << y << endl;
        cout << dfs(s, x, y, W, H) << endl;
    }

    return 0;
}
/*
6 9
....#.
.....#
......
......
......
......
......
#@...#
.#..#.
11 9
.#.........
.#.#######.
.#.#.....#.
.#.#.###.#.
.#.#..@#.#.
.#.#####.#.
.#.......#.
.#########.
...........
11 6
..#..#..#..
..#..#..#..
..#..#..###
..#..#..#@.
..#..#..#..
..#..#..#..
7 7
..#.#..
..#.#..
###.###
...@...
###.###
..#.#..
..#.#..
0 0

 */
