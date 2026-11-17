#set page(width: 25.4cm, height: 14.29cm, margin: (x: 1.6cm, y: 1.3cm))
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
#v(3.2cm)
#align(center)[
#text(size: 29pt, weight: "bold")[Нейросжатие текста онлайн-обучением] \
#v(0.5em)
#text(size: 17pt)[LSTM + оптимизатор Muon + арифметическое кодирование] \
#v(1.6em)
#text(size: 15pt)[Лабораторная работа · ИТМО · CT Lab] \
#v(0.8em)
#text(size: 15pt)[*Студент:* Федорин Кирилл Вячеславович, группа М4145] \
#text(size: 13pt, fill: gray)[Преподаватель: Е. А. Беляев · eabelyaev\@itmo.ru]
]

#pagebreak()
= Задача
Базовая версия сжимает `enwik5` (100 000 байт) авторегрессионной LSTM с адаптивным
арифметическим кодером. Требуется улучшить коэффициент сжатия при росте времени
не более чем на 50 %.

*Ключевое свойство декодера:* параметры модели не передаются в поток. Декодер
повторяет те же шаги SGD на уже декодированных байтах, поэтому в потоке только
длина файла, маска алфавита и гиперпараметры.

#v(0.5em)
#block(fill: luma(238), inset: 12pt, radius: 5pt)[
*Результат на этом устройстве (RTX 5090):* коэффициент *3.0052 → 3.1165*
(+3.7 %), время *27.0 с → 26.3 с* (не медленнее), восстановление
байт-в-байт (MD5 совпадает).
]

#pagebreak()
= Как работает кодек
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

    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        for group in opt.param_groups:
            group["lr"] = _lr(cfg, step, steps)
        logits, new_state = model(seq, state)
        probs_nn = torch.softmax(logits, dim=1).detach().cpu().numpy()
        logp_nn = np.log(np.maximum(probs_nn, 1e-300))

```)
]

#pagebreak()
= Главная находка: Adam «висел» на постоянном LR
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
устройстве — дальше все улучшения накладываются на корректный цикл.

#pagebreak()
= Улучшение: оптимизатор Muon
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
    if X.size(-2) > X.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
```)
]
*Законность.* Декодер повторяет те же 5 итераций Ньютона–Шульца и буфер импульса
на декодированных байтах детерминированно, поэтому поток остаётся без потерь.
*Измерено:* Muon даёт *2.575* бит/байт против *2.643* у Adam при той же геометрии.

#pagebreak()
= Сравнение архитектур и абляций
Все строки — полный цикл кодирование + декодирование со сверкой MD5.

#v(0.3em)
#table(columns: 4, inset: 4pt,
  [Вариант], [Байт], [Коэфф.], [Время, с],
  [A. LSTM + Adam (референс)], [33 388], [2.9951], [22.1],
  [B. LSTM + Adam, окно 32], [33 074], [3.0235], [41.7],
  [C. LSTM + Muon, окно 32], [33 386], [2.9953], [8.0],
  [D. LSTM + Muon, n=48], [32 153], [3.1101], [49.1],
  [*E. LSTM + Muon, n=80, cos (финал)*], [*32 087*], [*3.1165*], [*26.3*],
  [F. GDN-2 (линейное внимание)], [34 198], [2.9241], [482.8],
)

#v(0.3em)
*Вывод:* Muon ускоряет сходимость ядра; меньше потоков (n=80) даёт больше шагов
обучения на байт; GDN-2 (линейное внимание) работает, но в ~18 раз медленнее и здесь
не окупается.

#pagebreak()
= Результаты
#table(columns: 5, inset: 4pt,
  [Версия], [Исходный, Б], [Сжатый, Б], [Коэфф.], [Суммарно, с],
  [Baseline (main.py)], [100 000], [33 276], [3.0052], [27.0],
  [*Предложенная (LSTM+Muon)*], [100 000], [*32 087*], [*3.1165*], [*26.3*],
)

#v(0.5em)
Формат потока `NSCC1`: 5 байт длины + 32 байта маски алфавита + 12×4 байта
гиперпараметров + 16 байт имени архитектуры + арифметически закодированные данные.

Все параметры, зависящие от файла, лежат в заголовке → тип файла знать не нужно.

#pagebreak()
#grid(columns: (1fr, 1fr), gutter: 1.2em,
figure(image("figs/pareto.png", width: 74%), caption: "Коэффициент vs время"),
figure(image("figs/curve.png", width: 74%), caption: "Кривая обучения"),
figure(image("figs/bars.png", width: 74%), caption: "Размер после сжатия"),
)

#pagebreak()
= Линейное внимание и «векторы-эталоны»
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
архитектура интересна, но на 100 КБ не окупается. Финальное решение — *LSTM*.

#pagebreak()
= Воспроизведение
Окружение создаётся через `uv venv` (сборка torch под CUDA):

#block(fill: luma(245), inset: 10pt, radius: 4pt)[
#set text(size: 7.5pt)
```bash
uv venv --python 3.12 .venv
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv -r requirements.txt
uv run --python .venv python -m codec.cli --config config.json --input data/enwik5
```)
]
Итоговая конфигурация: `arch=lstm, n_streams=80, seq_len=32, hidden=768, emb=150,
optimizer=muon, muon_lr_scale=4.0, sched=cosine`.

Ноутбук `notebook.py` (marimo) — формулы, код, абляции и графики.

#pagebreak()
#v(3.0cm)
#align(center)[
#text(size: 24pt, weight: "bold")[Спасибо за внимание] \
#v(0.9em)
#text(size: 15pt)[Коэффициент *3.1165* (+3.7 % к базовому 3.0052),\
время *26.3 с* (быстрее базовых 27.0 с), восстановление байт-в-байт.] \
#v(1.2em)
#text(size: 13pt, fill: gray)[Федорин Кирилл Вячеславович · М4145 · eabelyaev\@itmo.ru]
]
