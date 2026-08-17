namespace Ahri {
void init_func() {
    int a = 0;
    int b(0);
    int c{0};
    int d = {0};
}
}  // namespace Ahri

int main(int argc, char* argv[]) {
    Ahri::init_func();

    return 0;
}
