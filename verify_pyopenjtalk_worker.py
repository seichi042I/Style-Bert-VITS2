#!/usr/bin/env python3
"""
pyopenjtalk_worker の起動と入出力検証スクリプト。
辞書登録（apply_word / delete_word）のテストを含む。
conda 環境 sbvits2 で実行: conda run -n sbvits2 python verify_pyopenjtalk_worker.py
"""
import sys
from pathlib import Path

# プロジェクトルートを path に追加
sys.path.insert(0, str(Path(__file__).resolve().parent))

import style_bert_vits2.nlp.japanese.pyopenjtalk_worker as pw
from style_bert_vits2.nlp.japanese.g2p import g2p
from style_bert_vits2.nlp.japanese.normalizer import normalize_text
from style_bert_vits2.nlp.japanese.user_dict import apply_word, delete_word, update_dict


def _run_dict_register_test() -> bool:
    """ユーザー辞書への登録・run_frontend での反映・削除を検証する。"""
    test_surface = "検証単語"
    test_pron = "ケンショウタンゴ"
    test_accent = 0
    sentence = "検証単語です。"
    word_uuid = None
    try:
        print("--- 辞書登録テスト ---")
        print(f"登録: surface={test_surface!r}, pronunciation={test_pron!r}, accent_type={test_accent}")
        word_uuid = apply_word(
            surface=test_surface,
            pronunciation=test_pron,
            accent_type=test_accent,
        )
        print(f"発行された UUID: {word_uuid}")

        parsed = pw.run_frontend(sentence)
        found = None
        for part in parsed:
            if part.get("string") == test_surface:
                found = part
                break
        if found is None:
            print(f"NG: run_frontend({sentence!r}) に {test_surface!r} が含まれませんでした。parsed={parsed}")
            return False
        if found.get("pron") != test_pron:
            print(f"NG: 期待 pron={test_pron!r}, 実際 pron={found.get('pron')!r}")
            return False
        print(f"OK: run_frontend で {test_surface!r} -> pron={found.get('pron')!r} を確認")
        return True
    finally:
        if word_uuid is not None:
            try:
                delete_word(word_uuid)
                print(f"登録した単語を削除しました (uuid={word_uuid})")
            except Exception as e:
                print(f"削除時にエラー (無視して続行): {e}")
    return False


def main() -> int:
    test_text = "こんにちは、今日は良い天気ですね。"
    print("=== pyopenjtalk_worker 入出力検証 ===\n")
    print(f"入力テキスト: {test_text!r}\n")

    # Worker 起動前は WORKER_CLIENT は None
    print(f"起動前 WORKER_CLIENT: {pw.WORKER_CLIENT}")
    pw.initialize_worker()
    print(f"起動後 WORKER_CLIENT: {type(pw.WORKER_CLIENT).__name__} (port={getattr(pw.WORKER_CLIENT, 'port', 'N/A')})\n")

    # dict_data/ を worker に適用（アプリ起動時と同様）
    update_dict()
    print("update_dict() 実行済み（default.csv + user_dict.json を適用）\n")

    # run_frontend 検証
    print("--- run_frontend 出力 ---")
    parsed = pw.run_frontend(test_text)
    print(f"型: {type(parsed)}, 要素数: {len(parsed)}")
    for i, part in enumerate(parsed):
        print(f"  [{i}] {part}")
    print()

    # run_frontend 各要素の string / pron のみ抜粋（G2P で利用する形）
    print("--- 表層形・読み (string / pron) ---")
    for i, part in enumerate(parsed):
        s = part.get("string", "?")
        p = part.get("pron", "?")
        print(f"  [{i}] string={s!r}  pron={p!r}")
    print()

    # make_label 検証（run_frontend の戻り値をそのまま渡す）
    print("--- make_label 出力（先頭5行）---")
    labels = pw.make_label(parsed)
    print(f"型: {type(labels)}, 要素数: {len(labels)}")
    for line in labels[:5]:
        print(f"  {line[:80]}{'...' if len(line) > 80 else ''}")
    print()

    # g2p による音素列取得テスト（平文 -> normalize_text -> g2p -> phones）
    print("--- g2p 音素列取得テスト ---")
    g2p_plain = "戦力外"
    norm_for_g2p = normalize_text(g2p_plain)
    phones, tones, word2ph = g2p(
        norm_for_g2p, use_jp_extra=True, raise_yomi_error=False
    )
    print(f"入力平文: {g2p_plain!r}")
    print(f"正規化後: {norm_for_g2p!r}")
    print(f"音素列 (phones): len={len(phones)}")
    print(f"  {phones}")
    print(f"アクセント (tones): len={len(tones)}, sum(tones)={sum(tones)}")
    print(f"word2ph: len={len(word2ph)}, sum={sum(word2ph)}")
    if len(phones) == 0 or phones[0] != "_" or phones[-1] != "_":
        print("NG: g2p の戻り値は先頭・末尾が _ の音素列である必要があります")
        pw.terminate_worker()
        return 1
    if len(phones) != len(tones) or sum(word2ph) != len(phones):
        print("NG: len(phones)==len(tones), sum(word2ph)==len(phones) である必要があります")
        pw.terminate_worker()
        return 1
    print("OK: g2p で音素列を取得し、形式を検証しました")
    print()
    g2p_plain = "せんりょくがい"
    norm_for_g2p = normalize_text(g2p_plain)
    phones, tones, word2ph = g2p(
        norm_for_g2p, use_jp_extra=True, raise_yomi_error=False
    )
    print(f"入力平文: {g2p_plain!r}")
    print(f"正規化後: {norm_for_g2p!r}")
    print(f"音素列 (phones): len={len(phones)}")
    print(f"  {phones}")
    print(f"アクセント (tones): len={len(tones)}, sum(tones)={sum(tones)}")
    print(f"word2ph: len={len(word2ph)}, sum={sum(word2ph)}")
    if len(phones) == 0 or phones[0] != "_" or phones[-1] != "_":
        print("NG: g2p の戻り値は先頭・末尾が _ の音素列である必要があります")
        pw.terminate_worker()
        return 1
    if len(phones) != len(tones) or sum(word2ph) != len(phones):
        print("NG: len(phones)==len(tones), sum(word2ph)==len(phones) である必要があります")
        pw.terminate_worker()
        return 1
    print("OK: g2p で音素列を取得し、形式を検証しました")
    print()

    # 辞書登録テスト（apply_word -> run_frontend 確認 -> delete_word）
    dict_ok = _run_dict_register_test()
    print()
    if not dict_ok:
        pw.terminate_worker()
        return 1

    # 別入力で短く再検証
    short_text = "はい"
    parsed_short = pw.run_frontend(short_text)
    print(f"短い入力 {short_text!r} -> run_frontend 要素数: {len(parsed_short)}")
    for part in parsed_short:
        print(f"  string={part.get('string')!r}  pron={part.get('pron')!r}")
    print()

    pw.terminate_worker()
    print("Worker 終了済み。検証完了.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
