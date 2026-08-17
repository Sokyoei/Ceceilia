#include "Ahri/Ceceilia/utils/charconvert.h"

/**
 * @brief GBK 转 UTF8
 * @param gbk_str
 * @return
 */
char* gbk_to_utf8(const char* gbk_str) {
#ifdef _WIN32
    // 先将 GBK 转换为宽字符
    int wcs_len = MultiByteToWideChar(CP_ACP, 0, gbk_str, -1, NULL, 0);
    wchar_t* wcs_str = (wchar_t*)malloc(wcs_len * sizeof(wchar_t));
    if (wcs_str == NULL) {
        perror("malloc");
        return NULL;
    }
    MultiByteToWideChar(CP_ACP, 0, gbk_str, -1, wcs_str, wcs_len);

    // 再将宽字符转换为 UTF-8
    int utf8_len = WideCharToMultiByte(CP_UTF8, 0, wcs_str, -1, NULL, 0, NULL, NULL);
    char* utf8_str = (char*)malloc(utf8_len);
    if (utf8_str == NULL) {
        perror("malloc");
        free(wcs_str);
        return NULL;
    }
    WideCharToMultiByte(CP_UTF8, 0, wcs_str, -1, utf8_str, utf8_len, NULL, NULL);

    free(wcs_str);
    return utf8_str;

#elif defined(__linux__)

    iconv_t cd;
    // 打开转换描述符，指定从 GBK 转换到 UTF-8
    cd = iconv_open("UTF-8", "GBK");
    if (cd == (iconv_t)-1) {
        perror("iconv_open");
        return NULL;
    }

    // 计算输入字符串长度
    size_t in_len = strlen(gbk_str);
    size_t out_len = in_len * 3;  // 一般情况下，UTF-8 编码可能会更长，乘以 3 是一个安全的估计
    char* out_str = (char*)malloc(out_len);
    if (out_str == NULL) {
        perror("malloc");
        iconv_close(cd);
        return NULL;
    }

    char* inbuf = (char*)gbk_str;
    char* outbuf = out_str;
    // 进行转换
    if (iconv(cd, &inbuf, &in_len, &outbuf, &out_len) == (size_t)-1) {
        perror("iconv");
        free(out_str);
        iconv_close(cd);
        return NULL;
    }

    // 关闭转换描述符
    iconv_close(cd);
    // 确保输出字符串以空字符结尾
    *outbuf = '\0';

    return out_str;
#endif
}

int main() {
    const char* gbk_str = "中文测试";
    char* utf8_str = gbk_to_utf8(gbk_str);
    if (utf8_str != NULL) {
        printf("GBK 字符串: %s\n", gbk_str);
        printf("转换后的 UTF-8 字符串: %s\n", utf8_str);
        free(utf8_str);
    }
    return 0;
}
