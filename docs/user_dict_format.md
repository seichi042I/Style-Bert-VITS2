# pyopenjtalk / OpenJTalk ユーザー辞書フォーマット

pyopenjtalk は OpenJTalk をラップしており、形態素解析には **MeCab** 互換の辞書（OpenJTalk では NAIST 日本語辞書 = IPADIC 系）を使用する。ユーザー辞書は **MeCab の CSV フォーマット** に従い、`mecab-dict-index` でバイナリ化してから読み込む。

## 公式・一次情報

- **MeCab 公式「単語の追加方法」**  
  https://taku910.github.io/mecab/dic.html  
  システム辞書・ユーザー辞書の追加手順と、CSV エントリのフォーマットが記載されている。

- **MeCab 公式（開発者・フォーマット説明）**  
  https://taku910.github.io/mecab/  
  出力フォーマットや設定の説明。

- **OpenJTalk**  
  https://open-jtalk.sourceforge.net/  
  NAIST 日本語辞書（IPADIC 系）を利用。辞書の CSV 構造は MeCab/ipadic と同一。

- **pyopenjtalk**  
  https://github.com/r9y9/pyopenjtalk  
  ユーザー辞書は `mecab_dict_index` で CSV をコンパイルし、`update_global_jtalk_with_user_dict(コンパイル済み.dic のパス)` で登録する。

---

## MeCab ユーザー辞書の CSV フォーマット（公式）

エントリは次の **13 列** の CSV（活用しない語の例）。

```
表層形,左文脈ID,右文脈ID,コスト,品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,原形,読み,発音
```

| 列 | 名前 | 説明 |
|----|------|------|
| 1 | 表層形 | 登録する単語の表記 |
| 2 | 左文脈ID | 単語を左から見たときの内部状態ID。空なら `mecab-dict-index` が自動付与 |
| 3 | 右文脈ID | 単語を右から見たときの内部状態ID。空なら自動付与 |
| 4 | コスト | 単語の出現しやすさ。小さいほど出現しやすい。似た品詞・単語と揃える |
| 5 | 品詞 | 品詞名 |
| 6 | 品詞細分類1 | 品詞細分類1 |
| 7 | 品詞細分類2 | 品詞細分類2 |
| 8 | 品詞細分類3 | 品詞細分類3 |
| 9 | 活用型 | 活用する語では「形容詞・イ段」など。活用しない語は `*` |
| 10 | 活用形 | 活用形。活用しない語は `*` |
| 11 | 原形 | 基本形の表記 |
| 12 | 読み | かな読み（カタカナ） |
| 13 | 発音 | 発音（カタカナ。多くの場合読みと同じ） |

### 公式の例（活用しない語）

```
工藤,1223,1223,6058,名詞,固有名詞,人名,名,\*,\*,くどう,クドウ,クドウ
メロンパン,1285,1285,4000,名詞,一般,\*,\*,\*,\*,メロンパン,メロンパン,メロンパン
```

左・右文脈ID とコストを空にすると、`mecab-dict-index` の `-m`（モデルファイル）指定時にコストを自動推定できる（公式 dic.html の「コストの自動推定機能」）。

### ユーザー辞書の作成手順（MeCab 公式）

1. UTF-8 などの CSV ファイル（例: `foo.csv`）に上記形式でエントリを追加する。
2. システム辞書があるディレクトリを `-d`、出力ファイルを `-u` に指定してコンパイルする。

   ```bash
   /usr/local/libexec/mecab/mecab-dict-index -d /usr/local/lib/mecab/dic/ipadic \
     -u foo.dic -f utf-8 -t utf-8 foo.csv
   ```

3. `mecabrc`（または `~/.mecabrc`）に `userdic = /path/to/foo.dic` を追加する。

pyopenjtalk では、コンパイル済み `.dic` を `pyopenjtalk.mecab_dict_index(csv_path, dic_path)` で生成し、`pyopenjtalk.update_global_jtalk_with_user_dict(dic_path)` で読み込む（本プロジェクトでは `style_bert_vits2.nlp.japanese.user_dict` がこの流れで辞書を更新している）。

---

## 本プロジェクトでの拡張（VOICEVOX 由来）

`dict_data/default.csv` および `user_dict` モジュールで出力する CSV は、上記 MeCab 13 列の後に **アクセント情報を付けた 2 列** を追加している。

- 列14: `アクセント型/モーラ数`（例: `0/3`, `4/5`）
- 列15: `アクセント結合ルール`（例: `*`, `C1`〜`C5`）

例:

```
Ｂｅｒｔ,,,8609,名詞,固有名詞,一般,*,*,*,Ｂｅｒｔ,バアト,バアト,0/3,*
担々麺,,,8609,名詞,固有名詞,一般,*,*,*,担々麺,タンタンメン,タンタンメン,3/6,*
```

左文脈ID・右文脈ID は空でもよく、`mecab_dict_index` がシステム辞書（OpenJTalk の NAIST 辞書）の `left-id.def` / `right-id.def` に従って扱う。品詞・品詞細分類・コスト候補・アクセント結合ルールは `part_of_speech_data`（VOICEVOX 由来）で定義されている。

---

## Function calling 用ツール（AI 穴埋めで CSV 追記）

各カラムを AI に穴埋めさせて 1 行ずつ CSV に追記するための function calling 用ツールを用意している。

- **スキーマ取得**: `style_bert_vits2.nlp.japanese.user_dict.function_calling.get_append_user_dict_tool_schema()`
  - OpenAI / Anthropic 等の `tools` にそのまま渡せる 1 件のツール定義（`type: "function"`）を返す。
- **実行関数**: `append_user_dict_entry(**kwargs)`
  - AI が埋めたパラメータを受け取り、`dict_data/ai_additions.csv` に 1 行追記する。戻り値は `{"success": True, "path": "...", "line": "..."}` または `{"success": False, "error": "..."}`。
- **必須パラメータ**: `surface`, `pronunciation`, `accent_type`
- **簡易指定**: `word_type` に `PROPER_NOUN` / `COMMON_NOUN` / `VERB` / `ADJECTIVE` / `SUFFIX` を指定すると、context_id や品詞細分類は省略可能。
- **マージ**: `update_dict()` 実行時に `default.csv` とユーザー辞書 JSON に続き、`ai_additions.csv` の内容もまとめてコンパイルされる。

使用例（疑似）:

```python
# クライアント側でツールを登録
tools = [get_append_user_dict_tool_schema()]

# AI が tool_calls で返した引数で実行
result = append_user_dict_entry(**ai_arguments)
if result["success"]:
    # 続けて update_dict() を呼ぶとコンパイルに反映される
    pass
```

---

## まとめ

- **ユーザー辞書のフォーマット**は **MeCab 公式の CSV 形式**（表層形, 左文脈ID, 右文脈ID, コスト, 品詞, 品詞細分類1〜3, 活用型, 活用形, 原形, 読み, 発音）。
- **公式ドキュメント**: MeCab の「単語の追加方法」  
  https://taku910.github.io/mecab/dic.html  
- pyopenjtalk では CSV を `mecab_dict_index` でコンパイルし、`update_global_jtalk_with_user_dict` で読み込む。本プロジェクトはその上で、アクセント用に 2 列を追加した形式を `default.csv` およびユーザー辞書コンパイル時に使用している。
