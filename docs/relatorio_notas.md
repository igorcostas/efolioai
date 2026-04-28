# Relatório — notas iniciais

- Estruturar a solução em módulos pequenos e testáveis.
- Separar domínio (`chess_pawn_mower`) de pesquisa (`search`).
- `board.py` preserva os 64 caracteres da instância, incluindo espaços, e converte entre a linha linear e a matriz 8×8.
- `state.py` modela a peça ativa, a origem dessa peça, a posição atual, o rei e os peões pretos restantes.
- `moves.py` aplica as regras do puzzle: capturas válidas por peça, bloqueios por ocupação e movimentos do rei para casas vazias ou peças brancas.
- `search/algorithms.py` implementa A* com timeout, ficando pronto para integrações futuras com BFS ou greedy.
- `io_utils/instances.py` lê `instancia_1.txt` a `instancia_10.txt` e devolve um estado inicial, usando tabuleiro vazio como fallback seguro quando o ficheiro não existe.
- `io_utils/csv_writer.py` garante o formato `ID;Tempo(ms);Solução` e normaliza a saída para 10 linhas.
- O limite temporal padrão é de 10 segundos por instância, materializado em `TIME_LIMIT_MS = 10000`.
- Para o relatório teórico, justificar:
  - `b` como fator médio de ramificação, maior quando existem muitas casas livres e várias peças brancas;
  - `d` como profundidade da primeira solução encontrada;
  - `m` como profundidade máxima imposta pelo limite de 100 ações;
  - heurística base: número de peões pretos restantes, admissível e barata de calcular.
- Justificar porque A* é preferível para custo mínimo, mantendo BFS apenas como referência teórica.
