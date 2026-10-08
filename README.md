# App Store ML — Midterm

Классификация **существующих приложений US App Store с ≥50 оценками**:
`high_rated = 1`, если `averageUserRating >= 4.5`, иначе 0.

Research question: **Can seven app metadata attributes distinguish highly rated (≥4.5) from
lower-rated existing US App Store apps with at least 50 user ratings at collection time?**

Это не проверка успеха приложения до запуска. Нейросеть, текстовые признаки и Streamlit оставлены для Final.

## Результат выполненного эксперимента

Реальный сбор 8 октября 2026 года (Asia/Qyzylorda): **2 029 приложений**, из них 1 651 класса 1
и 378 класса 0. Train/test: 1 623/406. Выполнены 3 pilot-запроса и 14 основных запросов
из списка 132; сбор остановился при достижении целевого размера.

| Модель | CV F1 ± SD | Test F1 | Test ROC-AUC |
| --- | --- | --- | --- |
| Dummy | 0.8974 ± 0.0009 | 0.8967 | 0.5000 |
| Decision Tree | 0.8779 ± 0.0068 | 0.8883 | 0.6777 |
| KNN | 0.8960 ± 0.0085 | 0.8964 | 0.6386 |
| SVM | 0.8989 ± 0.0027 | 0.9008 | 0.6200 |

По CV выбрана SVM. Прирост test F1 относительно Dummy — всего **+0.0041**.
SVM распознаёт только **7 из 76** приложений класса 0; balanced accuracy — 0.5415.
Высокий F1 обусловлен в том числе преобладанием класса 1 и не подтверждает сильную
способность различать классы. Статистическая значимость улучшения не заявляется.

[Полный эксперимент в GitHub Actions](https://github.com/Nura007/itune_ml/actions/runs/37685277008).
Числа выше получены из outputs/reports/*.csv; синтетические тесты к ним не относятся.

## Что реализовано

- Пробный сбор по трём запросам; основной сбор по 132 разнообразным ключевым словам.
- US/USD, ≤200 результатов/запрос, ≥3.2 секунды между запросами, кэш, ограниченные повторы.
- Исходные JSON, параметры запросов, время и SHA-256 сохраняются **локально**, в игнорируемом `data/raw/`.
- Дедупликация по `trackId`, фильтр ≥50, проверка типов/дат, журнал очистки.
- Семь признаков: цена, размер, жанр, возрастная категория, число языков, год выпуска, давность обновления.
- Постоянный stratified train/test 80/20, seed 42; пять CV-fold внутри train.
- DummyClassifier, Decision Tree, KNN, SVM; преобразования внутри Pipeline.
- F1 класса 1, ROC-AUC, precision, recall, balanced accuracy; среднее/SD CV и фиксированный test.
- Матрицы ошибок, ROC, EDA по train, реальные обезличенные ошибки, data card.
- Notebook и 9 слайдов на английском с заметками для выступления примерно на 9 минут.
- Автоматические проверки, отдельный запуск реального эксперимента.

## Источник и условия преподавателя

Владелец репозитория передал ответ преподавателя: использовать паузы, список запросов,
дедупликацию, исключить рейтинговые поля из признаков, не публиковать raw data и указать
promotional-content context Apple. Это записано в [config/source_approval.json](config/source_approval.json).

[Apple API documentation](https://performance-partners.apple.com/search-api) описывает Search API,
а [robots.txt](https://itunes.apple.com/robots.txt) содержит `Disallow: /search*`.
**Мы не утверждаем, что robots.txt разрешает сбор.** Согласование преподавателя не является
разрешением от Apple и не отменяет условия источника.

Не добавляйте raw JSON, описания, таблицы отдельных приложений, ID split, частные предсказания
или модели в этот публичный репозиторий. Автопубликация использует явный список разрешённых
сводных файлов.

## Локальный запуск с постоянным частным архивом

Нужен Python 3.11. Команды одинаковы на Windows/macOS/Linux, кроме активации venv.

```bash
git clone https://github.com/Nura007/itune_ml.git
cd itune_ml
git checkout codex/midterm
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
python -m appstore_ml.collect pilot
```

Проверьте `outputs/reports/pilot_audit.json`: реальные поля, пропуски, дубли, число подходящих приложений.
После приемлемого pilot:

```bash
python -m appstore_ml.collect main
python -m appstore_ml.prepare
python -m appstore_ml.run
```

Основной сбор останавливается при достижении **не менее 2 000** подходящих приложений.
Если после всех запросов осталось менее **1 500**, обучение прекращается с объяснением;
критерии исследования автоматически не ослабляются.

Повторный запуск использует тот же кэш. Изменившийся датасет не может незаметно заменить test.
Для нового сбора используйте отдельную директорию:

```bash
python -m appstore_ml.collect pilot --raw-dir work/run2/data/raw --report-dir work/run2/outputs/reports
python -m appstore_ml.collect main --raw-dir work/run2/data/raw --report-dir work/run2/outputs/reports
python -m appstore_ml.run --run-dir work/run2
```

Не запускайте два сборщика одновременно. Сохраняйте свою частную резервную копию `data/` и
`outputs/models/`; не снимайте gitignore.

## Результаты

После реального запуска:

- [Отчёт и выводы](outputs/reports/results.md)
- [Data card](outputs/reports/data_card.md)
- [CV](outputs/reports/cv_metrics.csv) и [test](outputs/reports/test_metrics.csv)
- [Notebook](notebooks/midterm.ipynb)
- [PowerPoint](outputs/midterm.pptx), [PDF](outputs/midterm.pdf), [предпросмотр](outputs/slides_contact.png)
- [Provenance](outputs/reports/provenance.json): версия кода, пакеты, даты, размеры split, fingerprint.

PDF и визуальный предпросмотр создаются в cloud workflow с LibreOffice. Локально основной
Python pipeline создаёт PPTX; для PDF откройте его в PowerPoint/LibreOffice.

Проверочные синтетические данные существуют только в tests и временных каталогах pytest.
Они не являются исследовательским датасетом и не используются в реальном эксперименте.

## GitHub Actions

`Protocol tests` проверяет границы фильтрации, отсутствие утечки, неизменность split и весь
pipeline на **явно помеченной синтетической fixture** без сетевого сбора.

`Approved Midterm experiment` запускается при изменении `config/collection_run.json`;
`stage: pilot` выполняет пробный сбор, `stage: full` — pilot и основной эксперимент.
Режим `stage: report` обновляет только оформление по сохранённым сводным результатам, без сбора и обучения.
Запуск выполняется только после тестов. Сводные результаты коммитятся в ту же ветку.

**Ограничение cloud run:** raw JSON, очищенные таблицы, split и модели существуют только на runner
до его удаления. Они не включаются в публичные artifacts. Для постоянного хранения исходных
данных и моделей выполните локальный запуск. Публичные сводные отчёты не позволяют
восстановить индивидуальные строки датасета.

## Структура

```text
config/                     протокол, 132 запроса, согласование источника
src/appstore_ml/             collect, prepare, features, split, pipelines, train, evaluate, report, run
tests/                      содержательные проверки без реального сетевого сбора
scripts/                    cloud orchestration и проверка публикации
notebooks/midterm.ipynb      объяснение и результаты
data/{raw,processed,splits}  частные данные; создаются запуском, не отслеживаются Git
outputs/reports/             публичные сводные таблицы, data card и выводы
outputs/figures/             графики
outputs/{private,models}/    частные файлы; не отслеживаются Git
docs/                       протокол, ограничения требований, вклад участников
app/                        только план Final
```

## Требования документов

Пути к `Midterm Rubric.docx` и `Project Guide.pdf` были указаны, но локальная среда не смогла
прочитать файлы. Реализация основана на подробных требованиях пользователя в чате.
[Матрица требований](docs/requirements.md) явно отмечает, что сверка с исходными документами
ещё не выполнена; полного соответствия непрочитанной rubric мы не заявляем.
