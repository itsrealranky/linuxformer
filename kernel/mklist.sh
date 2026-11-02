#!/bin/bash
# gen_init_cpio list for the initramfs: one busybox (FLAT) binary, its applet links, /dev/console.
# Usage: mklist.sh <busybox binary> <busybox.links> > initramfs.list
set -euo pipefail
BB=$(realpath "$1")
LINKS=$2
cat <<EOF
dir /dev 0755 0 0
nod /dev/console 0600 0 0 c 5 1
nod /dev/null 0666 0 0 c 1 3
dir /bin 0755 0 0
dir /sbin 0755 0 0
dir /usr 0755 0 0
dir /usr/bin 0755 0 0
dir /usr/sbin 0755 0 0
dir /proc 0755 0 0
dir /tmp 01777 0 0
dir /root 0700 0 0
file /bin/busybox $BB 0755 0 0
EOF
while read -r l; do
  [ "$l" = /bin/busybox ] && continue
  echo "slink $l /bin/busybox 0777 0 0"
done < "$LINKS"
