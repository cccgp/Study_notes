#include <iostream>
#include <vector>
using namespace std;

void dfs(vector<string>& s,int i,int j,int N,int M){
    if(!(i >= 0 && i < N && j >= 0 && j < M)) return;
    if(s[i][j] != 'W') return;
    s[i][j] = '.';
    for(int k = -1;k <= 1;k++){
        for(int l = -1;l <= 1;l++){
            dfs(s,i+k,j+l,N,M);
        }
    }
}
int main() {
    int N,M;
    cin >> N >> M;
    vector<string> strings(N);
    for(int i = 0;i < N;i++){
        cin >> strings[i];
    }
    int count = 0;
    for(int i = 0;i < N;i++){
        for(int j = 0;j < M;j++){
            if(strings[i][j] == 'W'){
                dfs(strings,i,j,N,M);
                count++;
            }
        }
    }
    cout << count << endl;
    return 0;
}
/*
10 12
W........WW.
.WWW.....WWW
....WW...WW.
.........WW.
.........W..
..W......W..
.W.W.....WW.
W.W.W.....W.
.W.W......W.
..W.......W.

 */
