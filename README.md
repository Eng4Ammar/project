MedVision AI
============

مشروع تعليمي باستخدام Streamlit وTensorFlow لتصنيف صور أشعة الصدر إلى COVID-19 وNormal وPneumonia.

محتويات المشروع
---------------

- `app.py`: واجهة التطبيق ومعالجة الصور والتحقق من نتائج النموذج.
- `model.keras`: النموذج المدرب المستخدم للتنبؤ.
- `requirements.txt`: اعتماديات متوافقة مع Python الحديث، مع تفضيل Python 3.11 للتشغيل المستقر.

التشغيل المحلي على Windows
--------------------------

يتطلب Python 3.11. افتح PowerShell داخل مجلد المشروع ونفّذ:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

النشر على Streamlit Community Cloud
-----------------------------------

1. ارفع `app.py` و`model.keras` و`requirements.txt` و`.python-version` و`README.md` إلى جذر مستودع GitHub.
2. أنشئ التطبيق من `share.streamlit.io`، واختر المستودع والفرع الصحيحين وملف `app.py`.
3. من إعدادات التطبيق اختر Python 3.11 ثم أعد تشغيل التطبيق. إصدار TensorFlow 2.15.1 متوافق مع النموذج المحفوظ. لا تستخدم Python 3.14 لهذا المشروع؛ سجل النشر أظهر عدم توافق في تحميل أوزان النموذج عليه.

لا ترفع ملفات البيانات الكبيرة أو دفاتر التدريب. ملف `.gitignore` يستثني دفتر الجلسة ومجلدات البيانات والبيئة المحلية. أوامر Git:

```powershell
git add app.py model.keras requirements.txt README.md .gitignore .python-version
git commit -m "Rebuild MedVision Streamlit app"
git push
```

ملاحظة مهمة عن ترتيب الفئات
--------------------------

ملف النموذج يحتوي على ثلاث درجات إخراج، لكنه لا يحفظ أسماء الفئات المرتبطة بكل درجة. التطبيق يستخدم الترتيب `COVID-19`, `Normal`, `Pneumonia` كما في النسخة السابقة. يجب مطابقة هذا الترتيب مع ترتيب الفئات الفعلي وقت تدريب النموذج قبل الاعتماد على أسماء النتائج؛ إعادة تدريب النموذج تتطلب ملفات الصور الأصلية.

التطبيق تعليمي فقط ولا يُستخدم للتشخيص الطبي.
