B="minlatitude=36&maxlatitude=41&minlongitude=25&maxlongitude=31"
for y in $(seq 2000 2025); do
 for h in "01-01 07-01" "07-01 01-01"; do set -- $h; s="$y-$1"; if [ "$2" = "01-01" ]; then e="$((y+1))-01-01"; else e="$y-07-01"; fi
  f=data/raw/isc/isc_${s}.txt; [ -s $f ] && continue
  curl -s -m 600 "https://www.isc.ac.uk/fdsnws/event/1/query?starttime=${s}&endtime=${e}&$B&minmagnitude=2.5&format=text&orderby=time-asc" -o $f; echo "ISC $s $(wc -l <$f)"; sleep 2
 done; done
for d in; do :
 curl -s -m 300 "https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime=$d-01-01&endtime=$e&$B&minmagnitude=3&orderby=time-asc" -o data/raw/usgs/usgs_$d.csv; echo "USGS $d $(wc -l < data/raw/usgs/usgs_$d.csv)"; done
cd data/raw/gcmt; curl -s -m 600 -O https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/jan76_dec20.ndk; ls -la
