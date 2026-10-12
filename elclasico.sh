#!/usr/bin/env bash
# El Clasico: you are FC Barcelona (PS controller), the AI is Real Madrid.
# Every play.sh option is spelled out here; edit any line to change it.
#   ~/projects/rl_project/elclasico.sh
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec "$ROOT/play.sh" \
  --level 11_vs_11_stochastic \
  --home "FC Barcelona" --home-short FCB \
  --away "Real Madrid" --away-short RMA \
  --home-players "Joan Garcia,Yamal,Lewandowski,Balde,Cubarsi,Araujo,Kounde,Pedri,De Jong,Gavi,Raphinha" \
  --away-players "Courtois,Mbappe,Rodrygo,Carreras,Huijsen,Rudiger,Alexander-Arnold,Bellingham,Tchouameni,Valverde,Vinicius" \
  --pitch normal \
  --power-assist 0.5 \
  --windup 2.0 \
  --speed 1.0 \
  "$@"
# Squad order is the formation slot order (GK first):
#   FCB: GK, RM, CF, LB, CB (left), CB (right), RB, CM (left), CM (deep), CM (right), LM
#   RMA: GK, ST, RW, LB, CB (left), CB (right), RB, CM (left), CM (deep), CM (right), LW
# Other switches (add to the end of the command line or above):
#   --level 11_vs_11_hard_stochastic   harder AI
#   --pitch dry|wet|long_grass         pitch conditions
#   --no-injury-time  --no-replays  --no-magnet  --upstream (original GRF)
