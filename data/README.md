# Dados da competição (NÃO versionados)

Descompacte aqui o bundle recebido após a inscrição, mantendo os nomes originais:

```
data/
├─ tactical/
│  ├─ BAR-ATH_tactical.mp4
│  ├─ BAR-ATH_tactical_atd.xml
│  ├─ BAR-ATH_tactical_atd.txt
│  ├─ BAR-ATH_tactical_homography.csv
│  └─ BAR-ATH_tactical_smart_tagging.xml
└─ broadcast/
   ├─ BAR-ATH_broadcast.mp4
   ├─ BAR-ATH_broadcast_atd.xml
   ├─ BAR-ATH_broadcast_atd.txt
   └─ BAR-ATH_broadcast_homography.csv
```

Para guardar em outro lugar, defina `MTAH_DATA_DIR` apontando para a pasta que contém `tactical/` e `broadcast/`.

Arquivos intermediários (parquet, caches) vão em `data/interim/`, também ignorado pelo git.

**Regras:** nada desta pasta vai para o git, para repositórios públicos ou para LLMs hospedados na nuvem (ChatGPT, Claude, Gemini etc.) — nem extratos. Ver `CONTRIBUTING.md`.
