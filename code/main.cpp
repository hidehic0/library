/**
 ベースのmain.cpp: https://github.com/hidehic0/library/blob/main/code/main.cpp
 oj-bundleで展開されたライブラリ: https://github.com/hidehic0/library_cpp
 ライセンスはどちらもUnlicenseです
 oj-bundleで展開されたのは/home/hidehic0/src以下にhttps://を付けるとgithubのソースに移動できます
 確認の際はそちらを見てください
**/
#include <bits/stdc++.h>
using namespace std;
#include <atcoder/all>
using namespace atcoder;

#ifdef ONLINE_JUDGE
#define dump(...)
#define CPP_DUMP_SET_OPTION(...)
#define CPP_DUMP_SET_OPTION_GLOBAL(...)
#define CPP_DUMP_DEFINE_EXPORT_OBJECT(...)
#define CPP_DUMP_DEFINE_EXPORT_ENUM(...)
#define CPP_DUMP_DEFINE_EXPORT_OBJECT_GENERIC(...)
#else
#include <cpp-dump/cpp-dump.hpp>
#define dump(...) cpp_dump(__VA_ARGS__)
#endif

#include "templates/alias.hpp"
#include "templates/macro.hpp"

int main() {}
