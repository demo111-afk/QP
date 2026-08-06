Option Explicit

Dim fso, shell, baseDir, pythonwPath, uiPath, command
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

baseDir = fso.GetParentFolderName(WScript.ScriptFullName)
pythonwPath = fso.BuildPath(baseDir, ".venv\Scripts\pythonw.exe")
If Not fso.FileExists(pythonwPath) Then
    pythonwPath = fso.BuildPath(baseDir, "venv\Scripts\pythonw.exe")
End If

If Not fso.FileExists(pythonwPath) Then
    MsgBox "Virtual Environment not found." & vbCrLf & vbCrLf & _
        "Initialize it once in PowerShell:" & vbCrLf & vbCrLf & _
        "py -m venv .venv" & vbCrLf & _
        ".venv\Scripts\python.exe -m pip install -r requirements.txt" & vbCrLf & _
        ".venv\Scripts\python.exe -m playwright install chromium", _
        vbCritical, "QP Copilot"
    WScript.Quit 1
End If

uiPath = fso.BuildPath(baseDir, "ui_app.pyw")
command = """" & pythonwPath & """ """ & uiPath & """"
shell.CurrentDirectory = baseDir
shell.Run command, 0, False
