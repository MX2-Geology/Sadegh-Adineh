B="minlatitude=36&maxlatitude=41&minlongitude=25&maxlongitude=31"
for y in "$@"; do for s in "$y-01-01 $y-07-01" "$y-07-01 $((y+1))-01-01"; do set -- $s
  f=data/raw/isc/isc_${1}.txt; [ -s $f ] && continue
  curl -s -m 900 "https://www.isc.ac.uk/fdsnws/event/1/query?starttime=$1&endtime=$2&$B&minmagnitude=2.5&format=text&orderby=time-asc" -o $f.part && mv $f.part $f; echo "ISC $1 $(wc -l <$f)"; done; done
