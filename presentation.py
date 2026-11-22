"""Презентация (слайды) лабораторной работы — marimo.

Просмотр:   .venv/bin/marimo edit presentation.py
PDF-слайды: .venv/bin/marimo export pdf presentation.py -o presentation.pdf \
                --as=slides --no-include-inputs

Каждый содержательный cell — один слайд. Служебный cell без вывода
пропускается при экспорте (--no-include-inputs).
"""
import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import pathlib

    import marimo as mo

    R = pathlib.Path(__file__).parent
    res = {r["name"]: r for r in json.loads((R / "final_results.json").read_text())}
    base = json.loads((R / "bench_baseline/results.json").read_text())
    N = 100_000
    ref_bytes = (R / "bench_baseline/data/enwik5_compressed.dat").stat().st_size
    ref_ratio = N / ref_bytes
    ref_time = base["time_spent"]

    def nfmt(x):
        return f"{x:,}".replace(",", "\u2009")

    def table(head, rows):
        lines = ["| " + " | ".join(head) + " |",
                 "|" + "|".join(["---"] * len(head)) + "|"]
        return "\n".join(lines + rows)

    def row(label, b, ratio, t):
        return f"| {label} | {nfmt(b)} | {ratio:.3f} | {t:.1f} |"

    final = res["G_LLM_GPT2_prior_fast"]
    lstm = res["E_LSTM_Muon_n80_s32_cosine_FINAL"]
    gdn = res["F_GDN_Muon_n48_s32"]

    S = dict(R=R, res=res, N=N, ref_bytes=ref_bytes, ref_ratio=ref_ratio,
             ref_time=ref_time, nfmt=nfmt, table=table, row=row,
             final=final, lstm=lstm, gdn=gdn)
    return (mo, S)


@app.cell
def _(mo):
    mo.md(
        """
# Нейросжатие текста

## Онлайн-модель и GPT-2 приор

**Федорин Кирилл Вячеславович**, группа М4145 \\
Преподаватель: Е. А. Беляев \\
Лабораторная работа, ИТМО, CT Lab

`github.com/Kiraprint/lstmcompressor_rsi`
"""
    )
    return


@app.cell
def _(S, mo):
    mo.md(
        f"""
## Задача

- Улучшить **коэффициент сжатия на ≥ 10 %** при росте времени **не более 50 %**.
- Бейзлайн `main.py` (RTX 5090): **{S['ref_ratio']:.4f}**,
  {S['nfmt'](S['ref_bytes'])} байт из {S['nfmt'](S['N'])} за
  **{S['ref_time']:.1f} с**.
- Цель: коэффициент **≥ 3.32**, время **≤ 40.5 с**.
- Тип файла **неизвестен**: все зависящие от файла гиперпараметры — в битовый поток.
- Демонстрация на `data/enwik5`, восстановление **байт-в-байт** (MD5).
"""
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
## Идея: языковая модель как приор

Литература из задания: `arXiv:2407.07723`, `bellard.org/nncp` — «тот же принцип,
но в качестве модели используется LLM».

**Схема:** файл → токены байтового BPE → вероятности следующего токена GPT-2 →
арифметический кодер Наюки (тот же, что в базовой версии).

Онлайн-модель из бейзлайна заменяется **замороженным GPT-2**: не нужно дообучать
веса на 100 КБ, модель уже знает язык.
"""
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
## Почему это легально по условию задачи

Правило: *«все гиперпараметры, **полученные предобработкой входного файла**,
должны быть в битовом потоке»*.

- Веса GPT-2 и токенизатор — **фиксированные константы алгоритма**, как
  арифметический кодер или архитектура сети. Они не получаются из файла.
- В потоке только **длина файла** и **число токенов** (заголовок 22 байта).
- Токенизатор **обратим для произвольных байтов**: Latin-1, все 256 значений
  проходят (проверено), поэтому тип файла действительно неизвестен.

Граница: BPE, выученный **на самом файле**, пришлось бы класть в поток (2–8 КБ) —
этот вариант проигрывает и здесь не используется.
"""
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
## Ключевой инженерный трюк: батч сегментов

Наивно: один проход GPT-2 на токен → **16.8 мс/токен** → ~450 с на файл.

**Решение:** поток режется на **независимые сегменты по 1000 токенов** (свой
KV-кэш у каждого), и все **27 сегментов считаются одним GPU-батчем** синхронно,
позиция за позицией.

- Вероятности те же (кодируемый кадр отличается на 1 байт выравнивания).
- Последовательно: **170.6 с** → батчем: **28.7 с** (≈20×).
"""
    )
    return


@app.cell
def _(S, mo):
    mo.md(
        "## Результаты на enwik5 (RTX 5090)\n\n"
        + S["table"](
            ["Версия", "Сжатый, Б", "Коэфф.", "Время, с"],
            [
                S["row"]("Бейзлайн (main.py)", S["ref_bytes"], S["ref_ratio"], S["ref_time"]),
                S["row"]("**GPT-2 приор (финал)**", S["final"]["bytes"],
                        S["final"]["ratio"], S["final"]["total_s"]),
                S["row"]("LSTM+Muon (self-contained)", S["lstm"]["bytes"],
                        S["lstm"]["ratio"], S["lstm"]["total_s"]),
            ],
        )
        + "\n\nВосстановление байт-в-байт, MD5 совпадает. "
        f"**+{100 * (S['final']['ratio'] / S['ref_ratio'] - 1):.0f} %** к коэффициенту "
        f"при времени **+{100 * (S['final']['total_s'] / S['ref_time'] - 1):.1f} %**."
    )
    return


@app.cell
def _(S, mo):
    mo.vstack([
        mo.md("## Коэффициент против времени"),
        mo.image(str(S["R"] / "figs" / "pareto.png"), width=620),
    ])
    return


@app.cell
def _(S, mo):
    mo.hstack([
        mo.vstack([mo.md("## Кривая обучения"), mo.image(str(S["R"] / "figs" / "curve.png"), width=330)]),
        mo.vstack([mo.md("## Размер файла"), mo.image(str(S["R"] / "figs" / "bars.png"), width=330)]),
    ])
    return


@app.cell
def _(S, mo):
    mo.md(
        f"""
## Самодостаточный вариант: онлайн LSTM + Muon

Кодек без внешних параметров — дообучает LSTM прямо по сжимаемому файлу.

- **{S['lstm']['ratio']:.4f}** (+{100 * (S['lstm']['ratio'] / S['ref_ratio'] - 1):.1f} %),
  {S['nfmt'](S['lstm']['bytes'])} байт, **{S['lstm']['total_s']:.1f} с**.
- Что дало прирост: корректное косинусное расписание LR (в эталоне оно было
  объявлено, но **не применялось**), оптимизатор **Muon** на рекуррентных
  матрицах, окно BPTT 32 и n=80.
- Это строгое решение «все параметры выучены из потока».
"""
    )
    return


@app.cell
def _(S, mo):
    mo.md(
        f"""## Что не сработало (отрицательные результаты)

| Подход | Коэфф. | Время, с | Вывод |
|---|---|---|---|
| GDN-2 (лин. внимание) | {S['gdn']['ratio']:.3f} | {S['gdn']['total_s']:.1f} | в 12× медленнее |
| GLA / SSM | 2.820 | 47.7 | хуже LSTM |
| Mamba-2 (selective) | 2.019 | 130.3 | вдвое хуже |
| Transformer | 3.77 bpb | 54 | слеп к long-range |
| Глубокий стек 2–6 слоёв | 2.62–4.87 bpb | 28–56 | монотонно хуже |
| Свой BPE в потоке | 2.49–2.90 | — | оверхед таблицы |
"""
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
## Проверка на контаминацию

GPT-2 предобучен на веб/вики-тексте, поэтому enwik5 может быть частично заучен:
teacher-forced **0.749 бит/байт**.

**Отложенные фрагменты enwik8** (смещения 50 МБ и 90 МБ, почти наверняка не
виденные):

- 0.83 и 0.87 бит/байт → коэффициент **9.2–9.7**.
- По-прежнему ~3× бейзлайна ⇒ результат **устойчив** к контаминации.
"""
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
## Воспроизведение

```bash
bash run.sh                    # онлайн LSTM+Muon (self-contained)
bash run.sh data/enwik5 --llm  # GPT-2 приор (финальное решение)
```

- Ноутбук анализа: `notebook.py` (marimo).
- Слайды: `presentation.py` (этот файл), экспорт — `marimo export pdf ... --as=slides`.
- Отчёт: `make_report.py` → `report.pdf`.
- Репозиторий: `github.com/Kiraprint/lstmcompressor_rsi`.
"""
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
# Спасибо за внимание

**GPT-2 приор: коэффициент 7.018** (в 2.3× от бейзлайна), 28.7 с — в бюджете.

Self-contained LSTM+Muon: 3.1165 (+3.7 %), 26.3 с.

Федорин Кирилл Вячеславович · М4145
"""
    )
    return


if __name__ == "__main__":
    app.run()
