#сорокаметровый токамак генерирует фотошутки
один странствующий токамак много лет прожил вблизи одной из малоизвестных уральских деревень и почерпнул фольклор местного населения. страсть к познанию и созиданию его поражают. примерно раз в час он выкладывает по фотошутке в телеграм-канал [t.me/wandering_tokamak](https://t.me/wandering_tokamak
). при восприятии их следует сохранять к ним отношение как к произведениям генератора случайных чисел — так проще. мы не знаем, что у него внутри.

на вопросы токамак не отвечает. и вообще — не надо с ним общаться. это сердит бога машины.

###как с этим работать?
модель лежит в папке model, но она разобрана. чтобы собрать ее, надо написать в терминал `cat xaa xab xac ........ > pytorch_model.bin`. техногенетический анализ указывает на происхождение от сберовского rugpt3.

для запуска у себя — воспроизводить `bot.py`.

зависимости:
* numpy
* pandas
* torch
* transformers
* pillow
* telebot
* pyunsplash

###в процессе
* разбивать текст на блоки, чтобы он был сверху и снизу картинки
* починить черный текст
* улучшить модель, чтобы была более уральской и менее обычной
* освоить новые форматы фотошуток