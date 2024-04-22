# Ceceilia

```mermaid
graph LR
  A[a.c] --> |-E 预处理| B[a.i];
  B --> |-S 汇编| C[a.s];
  C --> |-c 编译| D[a.o];
  D --> |链接| E[a.exe];
```
