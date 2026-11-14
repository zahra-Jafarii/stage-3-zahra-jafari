# راهنمای پروژه مرحله ۳: تصمیم پرهزینه

این پروژه با داده `Give Me Some Credit` مدلی می‌سازد که **هزینه کل تصمیم‌ها** را کم کند.
داده از دیتابیس خوانده می‌شود و با پکیج مرحله ۱ بارگذاری می‌شود.

## ۱. نصب

```bash
python -m venv .venv
pip install -e ../stage-1-zahra-jafari
pip install -e .
pip install -r requirements.txt
$env:DATABASE_URL="همان آدرس دیتابیس مرحله ۲"
```
* برای استفاده از پکیج‌های مراحل قبل 2 روش وجود دارد:(در مراحل نصب بالا از روش اول استفاده کردیم.)

روش اول: پوشه‌ها روی دیسک‌اند دراین حالت ساختار پوشه ها باید به شکل زیر باشد:
```bash
projects/
├── stage-1-zahra-jafarii/
├── stage-2-zahra-jafari/
└── stage-3-zahra-jafari/    ← از اینجا دستور را بزن
```
روش دوم: از GitHub هر دو ریپو را کنار هم clone می‌کنیم


## ۲. ترتیب اجرا

```bash
python -m credit_risk.prepare_input     # ۱) تغییر نام سرستون‌ها
python -m credit_risk.load_data         # ۲) کارخانه داده: اعتبارسنجی، قرنطینه، نوشتن در دیتابیس
jupyter notebook solution.ipynb         # ۳) Restart & Run All
python -m unittest discover -s tests -t .
python -m credit_risk.predict --input data/raw/new_file.csv
```

## ۳. وظیفه هر فایل

| فایل | وظیفه | کجا استفاده می‌شود |
|---|---|---|
| `settings.py` | seed، مسیرها، ماتریس هزینه ۱۲ به ۱، فهرست ستون‌ها، ثابت‌های دامنه‌ای | همه ماژول‌ها |
| `prepare_input.py` | کپی فایل Kaggle با سرستون snake_case (مقدارها دست نمی‌خورند) | قبل از بارگذاری |
| `dataset.py` | `Contract` و `Dataset` برای پکیج مرحله ۱ | `load_data.py` و `predict.py` |
| `load_data.py` | `run_pipeline` روی فایل و نوشتن در schema `stage3` | یک‌بار قبل از نوتبوک |
| `sql/clean_credit.sql` | قالب پاک‌سازی در لایه دیتابیس: کدهای ۹۶ و ۹۸ به `NULL`، سقف برای outlier | `db.py` |
| `db.py` | engine، ساخت view، خواندن با SQLAlchemy | نوتبوک و `predict.py` |
| `splitting.py` | جدا کردن holdout و قفل یک‌بار مصرف `HoldoutVault` | نوتبوک بخش ۲ و ۱۴ |
| `missingness.py` | شواهد الگوی گم‌شدگی | نوتبوک بخش ۴ |
| `features.py` | ۶ ویژگی جدید، فقط محاسبه سطری | نوتبوک و `predict.py` |
| `preprocessing.py` | imputer، scaler، encoder؛ سه سیاست گم‌شدگی | `models.py` |
| `models.py` | دو خانواده مدل: `logistic` و `boosting` | نوتبوک بخش ۵ تا ۱۴ |
| `baseline.py` | قاعده‌های ساده و هزینه آن‌ها | نوتبوک بخش ۶ و ۱۴ |
| `costs.py` | هزینه، منحنی هزینه، آستانه بهینه، حساسیت | نوتبوک بخش ۷ و ۸ |
| `evaluation.py` | پیش‌بینی out-of-fold، متریک‌ها، پایداری، زیرگروه‌ها | نوتبوک بخش ۷، ۱۱، ۱۳، ۱۴ |
| `ablation.py` | حذف تک‌تک ویژگی‌ها و مقایسه | نوتبوک بخش ۵ و ۹ |
| `importance.py` | اهمیت جایگشتی | نوتبوک بخش ۱۲ |
| `leakage.py` | بازرسی مرز نشت و جدول ممیزی | نوتبوک بخش ۲ و ۱۰ |
| `artifacts.py` | ذخیره و بارگذاری مدل، آستانه و ستون‌ها | نوتبوک بخش ۱۴ و `predict.py` |
| `predict.py` | فایل تازه ← کارخانه ← دیتابیس ← همان کوئری ← پیش‌بینی | خط فرمان |

## ۴. خروجی‌ها

| خروجی | مسیر | چه چیزی را نشان می‌دهد |
|---|---|---|
| `manifest_run.json` | `data/curated/` | تعداد خوانده‌شده، پذیرفته‌شده و قرنطینه |
| `quality_report.json` | `data/curated/` | آمار هر ستون |
| جدول `stage3.credit_training` | دیتابیس | داده پذیرفته‌شده |
| جدول `stage3.credit_training_quarantine` | دیتابیس | ردیف‌های ردشده با دلیل |
| view `stage3.v_credit_clean` | دیتابیس | داده پاک‌سازی‌شده برای مدل |
| `holdout_access.log` | `data/curated/` | ثبت زمان باز شدن holdout (فقط یک خط) |
| `credit_model.joblib` | `models/` | مدل، آستانه، ستون‌ها |
| `predictions.csv` | `data/curated/` | احتمال و تصمیم برای فایل تازه |

## ۵. نقشه بخش‌های صورت مسئله

| بخش صورت مسئله | محل انجام |
|---|---|
| ۱ شناخت داده | نوتبوک بخش ۳ |
| ۲ گمشده‌ها | `missingness.py`، نوتبوک بخش ۴ و ۵ |
| ۳ مرز نشت‌ناپذیر | `preprocessing.py`، `leakage.py`، نوتبوک بخش ۲ و ۱۰ |
| ۴ ویژگی‌ها و ممیزی | `features.py`، `ablation.py`، نوتبوک بخش ۹ و ۱۰ |
| ۵ خط مبنا و مدل‌ها | `baseline.py`، `models.py`، نوتبوک بخش ۶ و ۷ |
| ۶ آستانه | `costs.py`، نوتبوک بخش ۸ |
| ۷ ارزیابی و پایداری | `evaluation.py`، نوتبوک بخش ۱۱ و ۱۴ |
| ۸ تفسیر و انصاف | `importance.py`، نوتبوک بخش ۱۲ و ۱۳، `report.md` |
| ۹ مهندسی و تست | `tests/`، `artifacts.py`، `predict.py` |

## ۶. خطاهای رایج

| علامت | علت محتمل |
|---|---|
| خروج با کد ۶ | بالا رفتن نسبت قرنطینه؛ احتمالاً مقدار `NA` به‌عنوان گمشده شناخته نشده |
| `RuntimeError: DATABASE_URL` | متغیر محیطی تنظیم نشده |
| خطای ستون در view | نام ستون‌های جدول با `sql/clean_credit.sql` نمی‌خواند |
| `Holdout was already opened once` | holdout دوباره باز شده؛ kernel را Restart کنید |
