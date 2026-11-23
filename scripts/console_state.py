"""Private Windows state directories usable with both halves of a UAC token."""
import ctypes
from ctypes import wintypes
from pathlib import Path


def prepare_windows_state_directory(path):
    """Grant the actual account access, independent of the token's default owner.

    Python's Windows mkdir(mode=0o700) grants OWNER RIGHTS. An elevated token
    can make Administrators the owner, locking out the same account when it
    later runs without elevation. Install an account-SID ACL atomically when
    creating the directory; never replace permissions on an existing one.
    """
    path = Path(path)
    if path.is_dir():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    security = ctypes.WinDLL('advapi32', use_last_error=True)

    class SECURITY_ATTRIBUTES(ctypes.Structure):
        _fields_ = [('nLength', wintypes.DWORD),
                    ('lpSecurityDescriptor', ctypes.c_void_p),
                    ('bInheritHandle', wintypes.BOOL)]

    signatures = (
        (kernel.GetCurrentProcess, [], wintypes.HANDLE),
        (kernel.CloseHandle, [wintypes.HANDLE], wintypes.BOOL),
        (kernel.LocalFree, [ctypes.c_void_p], ctypes.c_void_p),
        (kernel.CreateDirectoryW, [wintypes.LPCWSTR, ctypes.POINTER(SECURITY_ATTRIBUTES)], wintypes.BOOL),
        (security.OpenProcessToken, [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)], wintypes.BOOL),
        (security.GetTokenInformation, [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                       wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)], wintypes.BOOL),
        (security.ConvertSidToStringSidW, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)], wintypes.BOOL),
        (security.ConvertStringSecurityDescriptorToSecurityDescriptorW,
         [wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(wintypes.DWORD)], wintypes.BOOL),
    )
    for function, arguments, result in signatures:
        function.argtypes, function.restype = arguments, result

    token = wintypes.HANDLE()
    if not security.OpenProcessToken(kernel.GetCurrentProcess(), 0x0008, ctypes.byref(token)):
        raise ctypes.WinError(ctypes.get_last_error())
    sid_text = ctypes.c_void_p()
    descriptor = ctypes.c_void_p()
    try:
        size = wintypes.DWORD()
        if not security.GetTokenInformation(token, 1, None, 0, ctypes.byref(size)):
            error = ctypes.get_last_error()
            if error != 122:  # ERROR_INSUFFICIENT_BUFFER
                raise ctypes.WinError(error)
        buffer = ctypes.create_string_buffer(size.value)
        if not security.GetTokenInformation(token, 1, buffer, size, ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        # TOKEN_USER starts with SID_AND_ATTRIBUTES, whose first field is PSID.
        sid = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_void_p))[0]
        if not security.ConvertSidToStringSidW(sid, ctypes.byref(sid_text)):
            raise ctypes.WinError(ctypes.get_last_error())
        account = ctypes.wstring_at(sid_text)
        sddl = f'O:{account}D:P(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)(A;OICI;FA;;;{account})'
        if not security.ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl, 1, ctypes.byref(descriptor), None):
            raise ctypes.WinError(ctypes.get_last_error())
        attributes = SECURITY_ATTRIBUTES(ctypes.sizeof(SECURITY_ATTRIBUTES), descriptor, False)
        if not kernel.CreateDirectoryW(str(path), ctypes.byref(attributes)):
            error = ctypes.get_last_error()
            if error != 183 or not path.is_dir():  # A concurrent console may have created it.
                raise ctypes.WinError(error)
    finally:
        if descriptor:
            kernel.LocalFree(descriptor)
        if sid_text:
            kernel.LocalFree(sid_text)
        kernel.CloseHandle(token)
