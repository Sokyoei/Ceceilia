#include <tchar.h>
#include <windows.h>

#pragma comment(lib, "Urlmon.lib")

int main() {
    HRESULT hresult = URLDownloadToFile(NULL, _T("https://github.com/Sokyoei/data/blob/master/Ahri/Ahri.xml"),
                                        _T("Ahri.xml"), 0, NULL);
    if (hresult == S_OK) {
        MessageBox(NULL, _T("下载成功"), _T("this is title"), MB_OK);
    } else {
        MessageBox(NULL, _T("下载失败"), _T("this is title"), MB_OK);
    }
    return 0;
}
