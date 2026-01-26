#!/bin/bash
set -e

MOUNT_POINT="/opt/dlami/nvme"
RAID_DEVICE="/dev/md0"

echo "=== NVMe RAID0 Setup Script for trn2.48xlarge ==="

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root (use sudo)"
   exit 1
fi

# Find all NVMe devices (excluding root device)
NVME_DEVICES=$(lsblk -d -n -o NAME,TYPE | grep nvme | grep disk | awk '{print "/dev/"$1}' | grep -v nvme0n1 || true)
NVME_COUNT=$(echo "$NVME_DEVICES" | grep -c nvme || echo 0)

echo "Found $NVME_COUNT NVMe devices:"
echo "$NVME_DEVICES"

if [[ $NVME_COUNT -lt 1 ]]; then
    echo "No additional NVMe devices found to configure."
    exit 1
fi

# Check if already mounted
if mountpoint -q "$MOUNT_POINT" 2>/dev/null; then
    echo "$MOUNT_POINT is already mounted."
    df -h "$MOUNT_POINT"
    exit 0
fi

# Check if RAID device already exists
if [[ -e "$RAID_DEVICE" ]]; then
    echo "RAID device $RAID_DEVICE already exists."
else
    echo "Creating RAID0 array with $NVME_COUNT devices..."

    # Stop any existing RAID arrays on these devices
    for dev in $NVME_DEVICES; do
        mdadm --zero-superblock "$dev" 2>/dev/null || true
    done

    # Create RAID0 array
    mdadm --create "$RAID_DEVICE" \
        --level=0 \
        --raid-devices=$NVME_COUNT \
        $NVME_DEVICES

    echo "RAID0 array created successfully."

    # Format with ext4
    echo "Formatting $RAID_DEVICE with ext4..."
    mkfs.ext4 -F "$RAID_DEVICE"
fi

# Create mount point
echo "Creating mount point $MOUNT_POINT..."
mkdir -p "$MOUNT_POINT"

# Mount the RAID device
echo "Mounting $RAID_DEVICE to $MOUNT_POINT..."
mount "$RAID_DEVICE" "$MOUNT_POINT"

# Set permissions
chown ubuntu:ubuntu "$MOUNT_POINT"
chmod 755 "$MOUNT_POINT"

# Show result
echo ""
echo "=== Setup Complete ==="
df -h "$MOUNT_POINT"
echo ""
echo "NVMe storage is now available at $MOUNT_POINT"
