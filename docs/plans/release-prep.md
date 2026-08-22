# План: подготовка репо к публичному релизу (до Reddit)

Дата: 2026-08-22. Источник: независимое ревью репозитория. Решения Кирилла:
порог детекции и числа не трогать (правило заморожено до запуска), панель уже
пересужена; остаются безопасность RunPod-раннеров и документация.

## Шаги

- [x] `control_server.py`: токен-защищённый статический сервер вместо
  публичного `python3 -m http.server`; токен на запуск, передаётся в Pod через
  env, клиент шлёт `Authorization: Bearer`. Тесты `tests/test_control_server.py`.
- [x] Watchdog: ключ RunPod через `EnvironmentFile=` (0600), не в argv
  `systemd-run`; `delete --env-file` удаляет файл.
- [x] `scripts/setup-runpod-v2-guard.sh`: помечен как operator-only, пути
  через env.
- [x] ruff clean; `requirements.txt`, `requirements-dev.txt`, `CITATION.cff`,
  `THIRD_PARTY_NOTICES.md`.
- [x] `README.md` под текущий эксперимент; `AGENTS.md` без stdlib-only;
  `results/README.md` карта canonical vs intermediate артефактов.
- [ ] Merge `codex/advanced-paraphrase-v2` → `main` (ff), тег `v0.2.0`, push.
- [ ] Сайт: ссылки статьи на тег.

Отвергнуто: перенос 58 промежуточных results в `results/archive/` (ломает
sha-ссылки provenance между артефактами); вместо этого карта. Переписывание
git-истории ради e-mail автора (решение оператора: дальше noreply).
