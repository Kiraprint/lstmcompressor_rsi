import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # LLM-prior codec: арифметическое кодирование с замороженным GPT-2

    Файл режется байтовым BPE на токены, и каждый токен кодируется
    арифметическим кодером, у которого распределение берётся из **GPT-2**.
    Веса и токенизатор — фиксированные константы алгоритма, в битовый поток
    не передаются: в кадре лежат только длина файла, число токенов и флаг
    режима токенизации.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Почему кодер требует полного словаря

    Nayuki-кодер на каждом символе читает из таблицы накопленных частот
    ровно три значения:

    $$
    \text{total} = c[-1], \qquad
    \text{lo} = c[s-1] \ (\text{или } 0),\qquad
    \text{hi} = c[s].
    $$

    Значит по сети нужно гонять не строку из 50 257 частот, а два числа.
    Инвариант, который делает кодирование точным:

    $$
    c_j = \left\lfloor \mathrm{SCALE}\cdot\sum_{i\le j} p_i \right\rfloor
    \quad\Longrightarrow\quad c[-1] = \text{SCALE}.
    $$

    Если $c[-1]\ne \mathrm{SCALE}$, интервалы получаются слишком узкими, коэффициент
    раздувается до ~9, и кадр не читается.
    """)
    return


@app.cell
def _(mo):
    import json

    with open("final_results.json") as f:
        rows = json.load(f)
    keep = [r for r in rows if r.get("ok")]
    keep.sort(key=lambda r: r.get("compressed_bytes") or r.get("bytes") or 10**9)
    table = "| прогон | байт | коэфф. | bpb |\n|---|---|---|---|\n" + "\n".join(
        f"| {r['name']} | {r.get('compressed_bytes') or r.get('bytes')} "
        f"| {r['ratio']:.4f} | {r.get('bpb', float('nan')):.4f} |"
        for r in keep
    )
    mo.md("## 2. Записанные результаты\n\n" + table)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Живой кодек

    Загружаем GPT-2 и токенизируем enwik5. Токенов получается **26 625**
    на 100 000 байт, то есть 3.76 байта на токен.
    """)
    return


@app.cell
def _():
    import llm_codec_fast

    codec = llm_codec_fast.LLMCodecFast()
    data = open("data/enwik5", "rb").read()
    ids = codec.tk.encode(data.decode("utf8")).ids
    return codec, ids


@app.cell
def _(ids, mo):
    mo.md(f"""
    Токенов: **{len(ids)}**, байт на токен: **{len(open('data/enwik5','rb').read())/len(ids):.2f}**.
    Сегментов по 1000 токенов: **{(len(ids) + 999) // 1000}**.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Цена обрыва контекста

    Сегменты независимы, поэтому контекст обрывается каждые `seg` токенов.
    Меняйте длину сегмента: один прогон предсказания занимает ~12 с,
    результат — сколько бит на байт стоит такой выбор.

    Внутри сегмента токены идут **строго последовательно**; параллельность
    только по измерению сегментов.
    """)
    return


@app.cell
def _(mo):
    seg = mo.ui.dropdown(options=[256, 512, 1000, 1024], value=1000, label="длина сегмента (токенов)")
    seg
    return (seg,)


@app.cell
def _(codec, ids, seg):
    import math

    codec.seg = seg.value
    (lo, hi), segs = codec._precompute(ids)
    total_bits = 0.0
    for pos in range(max(len(s) for s in segs)):
        if pos == 0:
            continue
        for b, s in enumerate(segs):
            if pos < len(s):
                total_bits += -math.log2(max((hi[pos][b] - lo[pos][b]) / (1 << 24), 1e-300))
    nbytes = len(open("data/enwik5", "rb").read())
    bpb_seg = total_bits / nbytes
    payload_seg = int(total_bits / 8) + 22
    n_segs = len(segs)
    return bpb_seg, n_segs, nbytes, payload_seg


@app.cell
def _(bpb_seg, mo, n_segs, nbytes, payload_seg, seg):
    mo.md(f"""
    | сегмент | сегментов | бит/байт | кадр | коэффициент |
    |---|---|---|---|---|
    | **{seg.value}** | **{n_segs}** | **{bpb_seg:.4f}** | **{payload_seg} Б** | **{nbytes / payload_seg:.3f}** |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Полный цикл: сжатие и распаковка

    Ниже — один реальный прогон: кадр, распаковка и сверка байт.
    Занимает ~27 с (кодирование + декодирование).
    """)
    return


@app.cell
def _():
    import hashlib

    # свой экземпляр с seg=1000: результат не зависит от выбора в разделе 4
    from llm_codec_fast import LLMCodecFast as _FullCodec

    _codec = _FullCodec()
    raw_bytes = open("data/enwik5", "rb").read()
    res = _codec.compress(raw_bytes)
    dec = _codec.decompress(res["frame"])
    back = dec["data"]
    dec_s = dec["decode_s"]
    ok = back == raw_bytes
    ratio_full = len(raw_bytes) / len(res["frame"])
    md5_full = hashlib.md5(back).hexdigest()
    return dec_s, md5_full, ok, ratio_full, raw_bytes, res


@app.cell
def _(dec_s, md5_full, mo, ok, ratio_full, raw_bytes, res):
    mo.md(f"""
    | | |
    |---|---|
    | байт было / стало | **{len(raw_bytes)} -> {len(res['frame'])}** |
    | коэффициент | **{ratio_full:.4f}** |
    | токенов | **{res['ntok']}** |
    | кодирование | **{res['encode_s']:.1f} с** |
    | декодирование | **{dec_s:.1f} с** |
    | восстановлено байт-в-байт | **{ok}** |
    | MD5 | `{md5_full}` |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Корректность для произвольных байтов

    Токенизатор байтовый, поэтому файл неизвестного типа тоже проходит:
    валидный UTF-8 идёт быстрым путём (`mode=0`), иначе — latin-1
    (`mode=1`), и распаковка восстанавливает те же байты.
    """)
    return


@app.cell
def _(codec):
    blob = bytes([0xFF, 0xFE, 0x00, 0x41, 0x80]) * 4
    r_blob = codec.compress(blob)
    bin_ok = codec.decompress(r_blob["frame"])["data"] == blob
    return bin_ok, blob, r_blob


@app.cell
def _(bin_ok, blob, mo, r_blob):
    mo.md(f"""
    Произвольные байты ({len(blob)} шт.): кадр **{len(r_blob['frame'])}** байт,
    восстановление **{bin_ok}**.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Воспроизведение

    ```bash
    bash run.sh data/enwik5 --llm     # полный цикл, ~27 с
    marimo edit llm_notebook.py       # этот ноутбук
    ```

    Ограничение честно: обрыв контекста каждые 1000 токенов стоит ~0.8 %
    размера относительно максимума GPT-2 (1024 позиции), а не параллельность.
    """)
    return


if __name__ == "__main__":
    app.run()
