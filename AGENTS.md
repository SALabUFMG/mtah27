# AGENTS.md

As instruções para agentes LLM deste repositório estão em **[CLAUDE.md](CLAUDE.md)** e valem para
qualquer agente (Codex, Gemini, Copilot, Cursor etc.). Leia-o inteiro antes de qualquer ação.

Resumo do que não pode ser esquecido:

1. **Nunca leia nem imprima dados da competição** (`data/`, vídeos, tracking, homografia, Smart
   Tagging, parquet, CSVs de submissão). Os termos proíbem enviá-los a LLMs na nuvem.
2. Nada amarrado a frames específicos, nenhum rótulo manual, não tunar nos 5 min públicos.
3. Não editar `external/mtah-official/`. Usar `uv`, nunca `pip`.
