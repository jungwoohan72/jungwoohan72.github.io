---
title: Conda 명령어
layout: post
categories: Blog
comments: true
---

1. 새로운 가상환경: conda create -m "env name" python="python version"
2. 가상환경 상속: conda create --name "new env name" --clone "old env name"
3. 가상환경에 설치된 package list: conda list
4. 가상환경 list: conda env list
5. 가상환경 삭제: conda remove --name "env name" --all
6. 패키지 다운로드: conda install -c conda-forge "package name"