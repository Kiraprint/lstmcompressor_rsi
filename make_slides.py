#!/usr/bin/env python3
"""Собирает PDF-презентацию (16:9, русский) из результатов и кода проекта."""
from __future__ import annotations

import json
from pathlib import Path

R = Path(__file__).parent
res = json.loads((R / "final_results.json").read_text())
by = {r["name"]: r for r in res}
base = by["baseline_reference"]
best = next(r for r in res if r["name"].endswith("FINAL"))
N = 100000
gain = (best["ratio"] / base["ratio"] - 1) * 100


def code(path: str, a: int, b: int) -> str:
    return "\n".join((R / path).read_text().splitlines()[a - 1:b])


loop_code = code("codec/core.py", 188, 196)
muon_code = code("codec/optim.py", 20, 26)
run_cmd = ("uv venv --python 3.12 .venv\n"
           "uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cu128\n"
           "uv pip install --python .venv -r requirements.txt\n"
           "uv run --python .venv python -m codec.cli --config config.json --input data/enwik5")


def table(head, rows):
    cell = ",\n  ".join(", ".join(f"[{c}]" for c in r) for r in [head] + rows)
    return f"#table(columns: {len(head)}, inset: 4pt,\n  {cell},\n)"


base_tbl = table(
    ["Версия", "Исходный, Б", "Сжатый, Б", "Коэфф.", "Суммарно, с"],
    [["Baseline (main.py)", "100 000", "33 276", "3.0052", "27.0"],
     ["*Предложенная (GPT-2 приор)*", "100 000", "*14 249*", "*7.018*", "*28.7*"],
     ["LSTM+Muon (self-contained)", "100 000", "32 087", "3.1165", "26.3"]])

abl_tbl = table(
    ["Вариант", "Байт", "Коэфф.", "Время, с"],
    [["A. LSTM + Adam (референс)", "33 388", "2.9951", "22.1"],
     ["B. LSTM + Adam, окно 32", "33 074", "3.0235", "41.7"],
     ["C. LSTM + Muon, окно 32", "33 386", "2.9953", "8.0"],
     ["D. LSTM + Muon, n=48", "32 153", "3.1101", "49.1"],
     ["*E. LSTM + Muon, n=80, cos (финал)*", "*32 087*", "*3.1165*", "*26.3*"],
     ["F. GDN-2 (линейное внимание)", "34 198", "2.9241", "482.8"]])

fig_pareto_90 = ('#figure(image("figs/pareto.png", width: 90%), '
              'caption: "Коэффициент vs время: все прогоны и зум на бюджет")')
fig_curve = 'figure(image("figs/curve.png", width: 100%), caption: "Кривая обучения")'
fig_bars = 'figure(image("figs/bars.png", width: 100%), caption: "Размер после сжатия")'

PREAMBLE = """#set page(width: 25.4cm, height: 14.29cm, margin: (x: 1.6cm, y: 1.3cm))
#set text(font: ("DejaVu Sans"), size: 12.5pt)
#set par(justify: true, leading: 0.78em)
#show heading.where(level: 1): set text(size: 21pt, weight: "bold")
#show table: set text(size: 10pt)
#show heading.where(level: 1): set block(below: 0.7em)

#set page(header: context {
  set text(size: 9pt, fill: gray)
  grid(columns: (1fr, 1fr), align: (left, right),
    [Нейросжатие текста · Федорин К. В., М4145],
    [стр. #counter(page).display()])
})
"""

slides = []

slides.append("""#v(3.2cm)
#align(center)[
#text(size: 29pt, weight: "bold")[Нейросжатие текста онлайн-обучением] \\
#v(0.5em)
#text(size: 17pt)[LSTM + оптимизатор Muon + арифметическое кодирование] \\
#v(1.6em)
#text(size: 15pt)[Лабораторная работа · ИТМО · CT Lab] \\
#v(0.8em)
#text(size: 15pt)[*Студент:* Федорин Кирилл Вячеславович, группа М4145] \\
#text(size: 13pt, fill: gray)[Преподаватель: Е. А. Беляев · eabelyaev\\@itmo.ru]
]""")

slides.append(f"""= Задача
Базовая версия сжимает `enwik5` (100 000 байт) авторегрессионной LSTM с адаптивным
арифметическим кодером. Требуется улучшить коэффициент сжатия при росте времени
не более чем на 50 %.

*Ключевое свойство декодера:* параметры модели не передаются в поток. Декодер
повторяет те же шаги SGD на уже декодированных байтах, поэтому в потоке только
длина файла, маска алфавита и гиперпараметры.

#v(0.5em)
#block(fill: luma(238), inset: 12pt, radius: 5pt)[
*Результат на этом устройстве (RTX 5090):* GPT-2-приор — коэффициент
*3.0052 → 7.018* (в 2.3× базового, +133 %), время *27.0 → 28.7 с* (в бюджете
+50 %). Self-contained LSTM+Muon — *3.1165* (+3.7 %), 26.3 с. Обе версии
восстанавливают байт-в-байт (MD5 совпадает).
]""")

slides.append(f"""= Как работает кодек
$
  p(x_1 ... x_N) = product_(t=1)^N p_theta (x_t | x_(<t)), quad
  L = sum_t -log_2 p_theta (x_t | x_(<t)) .
$
Кодек «платит» ровно $-log_2 p$ бит за символ, поэтому оптимизируем именно $L$.

*Три рычага снижения $L$:*
+ *Алфавит из файла* — 256-битная маска, без предположений о типе файла (155 из 256).
+ *Статистическое смешивание* — нейросеть учится медленно; смешиваем её с быстрыми
  order-1..6 и LZP-экспертами геометрической смесью $log p = sum_j a_j log p_j$,
  $a = op("softmax")(u)$ на симплексе.
+ *Быстрая сходимость ядра* — заменяем Adam на *Muon* и удлиняем окно BPTT.

#v(0.2em)
#block(fill: luma(245), inset: 8pt, radius: 4pt, width: 100%)[
#set text(size: 7.5pt)
```python
{loop_code}
```)
]""")

slides.append("""= Главная находка: Adam «висел» на постоянном LR
При воспроизведении базовой версии оказалось, что в нашем цикле *расписание
learning rate не применялось* — LR всё время оставался $5e-3$, тогда как референс
линейно снижает его до $1e-4$. Модель недоучивалась и проигрывала ~6 % по битам.

После включения $lr(k) = lr_"end" + (lr_"start" - lr_"end")(1 - k slash T)$
кривая обучения *совпала с референсом*:

#v(0.4em)
#table(columns: 3, inset: 4pt,
  [Шаг], [Референс, бит/байт], [Наш кодек],
  [256], [3.3227], [3.3253],
  [512], [2.9107], [2.9137],
  [768], [2.6619], [2.6656],
)

#v(0.4em)
Это дало *честную точку отсчёта*: референс $3.0052$, наш кодек $2.9951$ на том же
устройстве — дальше все улучшения накладываются на корректный цикл.""")

slides.append(f"""= Улучшение: оптимизатор Muon
Для рекуррентных матриц $W$ Muon хранит буфер импульса $B_t$ и применяет
*ортогонализованное* обновление (Ньютон–Шульц, 5 итераций):
$
  B_t = mu B_(t-1) + (1-mu) nabla_W L, quad
  W <- W - eta sqrt(max(m,n)/min(m,n)) thin U_t .
$
Adam масштабирует *поэлементно*, из-за чего несколько крупных весов подавляют
остальные; ортогонализация выравнивает все сингулярные направления.

#block(fill: luma(245), inset: 8pt, radius: 4pt)[
#set text(size: 7.5pt)
```python
{muon_code}
```)
]
*Законность.* Декодер повторяет те же 5 итераций Ньютона–Шульца и буфер импульса
на декодированных байтах детерминированно, поэтому поток остаётся без потерь.
*Измерено:* Muon даёт *2.575* бит/байт против *2.643* у Adam при той же геометрии.""")

slides.append(f"""= Сравнение архитектур и абляций
Все строки — полный цикл кодирование + декодирование со сверкой MD5.

#v(0.3em)
{abl_tbl}

#v(0.3em)
*Вывод:* Muon ускоряет сходимость ядра; меньше потоков (n=80) даёт больше шагов
обучения на байт; GDN-2 (линейное внимание) работает, но в ~18 раз медленнее и здесь
не окупается.""")

slides.append(f"""= Результаты
{base_tbl}

#v(0.5em)
Формат потока `NSCC1`: 5 байт длины + 32 байта маски алфавита + 12×4 байта
гиперпараметров + 16 байт имени архитектуры + арифметически закодированные данные.

Все параметры, зависящие от файла, лежат в заголовке → тип файла знать не нужно.""")

slides.append(f"""= Коэффициент против времени
Слева — все прогоны (время логарифмически), справа — зум на укладывающиеся в бюджет.
Пунктир — бейзлайн 3.005, зелёная — цель +10 % (3.306), оранжевая — бюджет 40.5 с.

{fig_pareto_90}""")

slides.append(f"""= Сходимость и размер файла
#grid(columns: (1fr, 1fr), gutter: 1.4em,
  {fig_curve},
  {fig_bars},
)""")

slides.append("""= Линейное внимание и «векторы-эталоны»
Пробовали заменить LSTM на *gated linear attention* с матричной памятью
$S_t in RR^(d_v times d_k)$: каждый столбец — вектор-эталон, а считывание
$S_t^T q_t$ — запрос-реконструкция:
$
  S_t = a_t S_(t-1) + k_t v_t^T, quad a_t = sigma(-"gate"(e_t)) in (0,1).
$
Вариант GDN-2 добавляет *стирание* (правило дельты), позволяя перезаписать
направление за один шаг:
$
  S_t = (I - beta_t k_t k_t^T) a_t S_(t-1) + beta_t v_t k_t^T .
$
Метрика: GDN-2 дал $2.924$ против $3.117$ у LSTM+Muon при времени $483$ с —
архитектура интересна, но на 100 КБ не окупается. Лучший онлайн-результат —
*LSTM+Muon*; абсолютный — *GPT-2 приор* (следующий слайд).""")

slides.append("""= Финальное решение: GPT-2 приор

Вместо онлайновой сети — *замороженный* GPT-2 как приор
(arXiv:2407.07723, bellard.org/nncp). Токены — обратимый байтовый BPE
(Latin-1, все 256 байт проходят). Логиты — в тот же арифметический кодер.

#block(fill: luma(245), inset: 10pt, radius: 4pt)[
#set text(size: 12.5pt)
Сегменты по 1000 токенов *независимы* (свежий KV-кэш) → все 27 сегментов
считаются одним GPU-батчем синхронно (~20× дешевле; разница кадров 1 байт).
]

#set text(size: 10.5pt)
#table(
  columns: (1.5fr, 1fr, 1fr, 1fr),
  inset: 3pt,
  [*Метод*], [*Байт*], [*Коэфф.*], [*Время (с)*],
  [Базовый (RTX 5090)], [33 276], [3.0052], [27.0],
  [*GPT-2 приор (батч)*], [*14 249*], [*7.018*], [*28.7*],
  [LSTM+Muon (self-contained)], [32 087], [3.1165], [26.3],
)
#set text(size: 12.5pt)
Байт-в-байт (MD5 совпал): *7.018* — в 2.3× выше базового (+133 %), в 2× выше
цели +10 %; время 28.7 с — в бюджете +50 % (+6.3 %). Модель и токенизатор —
*фиксированные константы алгоритма*, не получены из входного файла → правило
«параметры предобработки файла в потоке» выполнено (в потоке только длина и
число токенов). На held-out enwik8 (не контаминирован) ratio 9.2–9.7.""")

slides.append(f"""= Воспроизведение
Исходный код проекта: https://github.com/Kiraprint/lstmcompressor_rsi

Окружение создаётся через `uv venv` (сборка torch под CUDA):

#block(fill: luma(245), inset: 10pt, radius: 4pt)[
#set text(size: 7.5pt)
```bash
{run_cmd}
```)
]
Итоговая конфигурация: `arch=lstm, n_streams=80, seq_len=32, hidden=768, emb=150,
optimizer=muon, muon_lr_scale=4.0, sched=cosine`.

Ноутбук `notebook.py` (marimo) — формулы, код, абляции и графики.""")

slides.append(f"""#v(3.0cm)
#align(center)[
#text(size: 24pt, weight: "bold")[Спасибо за внимание] \\
#v(0.9em)
#text(size: 15pt)[*Финальное решение — GPT-2-приор: 7.018* (в 2.3× базового, +133 %), 28.7 с. \\
Self-contained LSTM+Muon: *3.1165* (+3.7 %), 26.3 с. \\
Восстановление байт-в-байт (MD5).] \\
#v(1.2em)
#text(size: 13pt, fill: gray)[Федорин Кирилл Вячеславович · М4145 · eabelyaev\\@itmo.ru]
]""")

doc = PREAMBLE + "\n\n#pagebreak()\n".join(slides) + "\n"
(R / "slides.typ").write_text(doc, encoding="utf-8")
print("slides.typ", len(doc), "bytes,", len(slides), "slides")
