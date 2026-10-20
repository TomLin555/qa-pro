# FCCL 銅箔 AI 檢測客服（Streamlit）

以 Streamlit 提供繁體中文客服對話介面，使用既有 100 組問答的 384 維神經網路語意向量尋找答案。支援聊天紀錄、常見問題、清除對話與參考問答。不需要生成式模型或付費 API 金鑰；回覆直接取自知識庫，沒有符合門檻的內容時會請使用者補充問題。每一則提問獨立檢索，不會根據前文推測省略的問題。

## 本機啟動

建議使用 Python 3.12。以獨立環境安裝網頁套件即可：

```powershell
python -m venv .venv-web
.\.venv-web\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-web\Scripts\python.exe -m streamlit run app.py
```

或使用 uv：

```powershell
uv run --no-project --with-requirements requirements.txt python -m streamlit run app.py
```

開啟 http://localhost:8501 。第一次提問會下載固定版本模型（約 135 MB），之後快取重用。模型以 CPU 執行，不需要 GPU；首次下載與冷啟動可能較久。不要使用 `python app.py` 啟動 Streamlit 網頁。

## 部署至 Streamlit Community Cloud

1. 解壓縮 `streamlit_upload.zip`，將內容上傳至一個 GitHub repository，保留 `.streamlit` 子目錄。
2. 在 [Streamlit Community Cloud](https://share.streamlit.io/) 登入，選擇 **Create app**，指定該 repository 與 branch。
3. **Main file path** 填入 `app.py`；在 **Advanced settings** 選擇 Python **3.12**。
4. 預設本機向量索引模式不需 Secrets，直接部署。若使用 Supabase，先依下節設定 Secrets。
5. 等待應用啟動，再測試原始提問、改寫提問及無關問題；首次提問需要等待模型下載。

部署包包含 `app.py`、`qa_service.py`、`build_index.py`、`requirements.txt`、`QA.csv`、`qa_index.npz`、`qa.sql`、`README.md` 與 `.streamlit` 設定範本。

**請使用部署包內容建立部署倉庫。** 原工作目錄的 `uv.lock` 與 `pyproject.toml` 是資料分析專案設定，未納入部署包；若直接上傳整個工作目錄，Community Cloud 可能優先採用 `uv.lock`，導致沒有安裝 Streamlit，或選到不符合的 Python 版本。不要將這兩個檔案放進部署入口的目錄。不要上傳 `.venv`、`.venv-web`、`.env` 或真實金鑰。

目前已完成可部署程式與打包，尚未連接您的 GitHub 或發佈到 Community Cloud。

## 選用 Supabase

先在 Supabase SQL Editor 執行 `qa.sql`，建立 `qa` 資料表與 `match_qa` RPC。接著在 Community Cloud 的 **Advanced settings → Secrets** 貼入：

```toml
QA_BACKEND = "supabase"
QA_MATCH_THRESHOLD = "0.45"
SUPABASE_URL = "https://YOUR_PROJECT.supabase.co"
SUPABASE_SERVICE_ROLE_KEY = "填入後端 service role 金鑰"
```

本機也可將 `.streamlit/secrets.toml.example` 複製為 `.streamlit/secrets.toml` 後填值。真實 `secrets.toml` 已列入 `.gitignore`，不要提交到 GitHub。也可以透過作業系統環境變數設定；`.env.example` 僅供參考，應用不會自動讀取 `.env`。

金鑰僅由伺服器讀取。現有 SQL 啟用了 RLS 且未開放前端讀取，因此此模式使用 service role。Supabase 查詢失敗會顯示服務暫時無法使用，不會默默切換資料來源。此模式仍以本機模型生成查詢向量，再呼叫 Supabase RPC 搜尋。

## 向量與資料維護

- 模型：`Xenova/paraphrase-multilingual-MiniLM-L12-v2`，固定 revision `2c4055b12046f11709e9df2c122e59ffbdc2f900`，使用 `onnx/model_quantized.onnx`。
- 384 維；文字分成每段最多 126 個內容 token，加上起訖 token，使用遮罩平均池化、依內容 token 數加權平均，再做 L2 正規化，與現有 `qa.sql` 相同。
- 文件輸入為 `question + 換行 + answer`；查詢輸入為使用者問題。使用餘弦相似度比對。
- `qa_index.npz` 擷取 `qa.sql` 既有向量；載入時核對資料摘要，避免 CSV 與向量錯位。
- `QA_MATCH_THRESHOLD` 預設 `0.45`，範圍 0 到 1。這是初始檢索門檻，尚未經正式客服測試集校準，相似度不是答案正確率。
- 本機模式對完全相同的問題直接回傳原答案，其餘使用語意檢索；Supabase 模式統一使用 RPC。
- 模型透過 `st.cache_resource` 共用；每位使用者的對話放在各自的 `st.session_state`，不存入共用模型或資料庫。
- 若修改問答，需先以相同模型與流程重新生成 `qa.sql`，再執行 `build_index.py` 更新索引。此工具只能匯出既有向量，不能為新問答生成向量。

```powershell
uv run --no-project --with-requirements requirements.txt python build_index.py
uv run --no-project --with-requirements requirements.txt python -m unittest discover -s tests -v
```

可透過環境變數 `QA_MODEL_DIR` 指定已有 `tokenizer.json` 與 `model_quantized.onnx` 的資料夾，以離線載入。必須使用上述固定 revision 的檔案。

知識庫保留來源文件的設備規格與操作說明，正式上線前應由產品負責人對照實際系統審核。對話僅保留於目前連線的工作階段，不提供跨連線的永久聊天紀錄。

參考：[Community Cloud 部署](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)、[相依套件優先順序](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)、[Secrets 設定](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management)。
