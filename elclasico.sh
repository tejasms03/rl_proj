#!/usr/bin/env bash
# El Clasico: you are FC Barcelona (PS controller), the AI is Real Madrid.
# Every play.sh option is spelled out here; edit any line to change it.
#   ~/projects/rl_project/elclasico.sh
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec "$ROOT/play.sh" \
  --level 11_vs_11_stochastic \
  --home "FC Barcelona" --home-short FCB \
  --away "Real Madrid" --away-short RMA \
  --home-kit barca_home \
  --away-kit madrid_home \
  --home-players "Joan Garcia:1:short01:darkblonde,Yamal:3:short02:black,Lewandowski:1:short01:darkblonde,Balde:4:short02:black,Cubarsi:1:short01:darkblonde,Araujo:2:short01:black,Kounde:4:short02:black,Pedri:2:short01:black,De Jong:1:medium01:darkblonde,Gavi:2:short01:black,Raphinha:3:short02:black" \
  --away-players "Courtois:1:short01:darkblonde,Mbappe:4:bald:black,Rodrygo:3:short02:black,Carreras:1:short01:darkblonde,Huijsen:1:short01:darkblonde,Rudiger:4:short02:black,Alexander-Arnold:3:short02:black,Bellingham:3:short02:black,Tchouameni:4:short02:black,Valverde:2:short01:darkblonde,Vinicius:4:short02:black" \
  --pitch normal \
  --power-assist 0.5 \
  --windup 2.0 \
  --speed 1.0 \
  "$@"
# Squad order is the formation slot order (GK first). Each entry is
# Name:skin:hair_style:hair_color (skin 1 lightest - 4 darkest).
#   FCB: GK, RM, CF, LB, CB (left), CB (right), RB, CM (left), CM (deep), CM (right), LM
#   RMA: GK, ST, RW, LB, CB (left), CB (right), RB, CM (left), CM (deep), CM (right), LW
# Other switches (add to the end of the command line or above):
#   --level 11_vs_11_hard_stochastic   harder AI
#   --pitch dry|wet|long_grass         pitch conditions
#   --home-kit / --away-kit  barca_home|madrid_home|madrid_away|red|sky|builtin
#   --no-injury-time  --no-replays  --no-magnet  --upstream (original GRF)
