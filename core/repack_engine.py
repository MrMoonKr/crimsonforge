"""Full repack pipeline with complete checksum chain management.

Implements the complete repack process:
1. Validate modified files
2. Create backup
3. Compress + encrypt modified data
4. Write to PAZ (16-byte aligned)
5. Update PAZ CRC in PAMT
6. Update file entry sizes in PAMT
7. Update PAMT self-CRC
8. Write PAMT CRC to PAPGT
9. Update PAPGT self-CRC
10. Verify all checksums
"""

import os
import struct
from dataclasses import dataclass
from typing import Optional, Callable

from core.checksum_engine import pa_checksum
from core.crypto_engine import encrypt
from core.compression_engine import compress
from core.pamt_parser import (
    parse_pamt, PamtData, PamtFileEntry,
    update_pamt_paz_entry, update_pamt_file_entry, update_pamt_self_crc,
)
from core.papgt_manager import (
    parse_papgt, PapgtData,
    get_pamt_crc_offset, update_papgt_pamt_crc, update_papgt_self_crc,
)
from core.backup_manager import BackupManager
from core.paz_write_utils import build_space_map, write_entry_payload
from utils.logger import get_logger
from utils.platform_utils import (
    get_file_timestamps, set_file_timestamps, atomic_write,
)

logger = get_logger("core.repack_engine")


@dataclass
class ModifiedFile:
    """A file to be repacked into the game archives."""
    data: bytes
    entry: PamtFileEntry
    pamt_data: PamtData
    package_group: str


@dataclass
class RepackResult:
    """Result of a repack operation."""
    success: bool
    files_repacked: int
    paz_crc: int
    pamt_crc: int
    papgt_crc: int
    backup_dir: str
    errors: list[str]


class RepackEngine:
    """Manages the full repack pipeline with checksum chain integrity."""

    def __init__(self, packages_path: str, backup_dir: str = ""):
        self._packages_path = packages_path
        self._backup_dir = backup_dir or os.path.join(packages_path, "..", "crimsonforge_backups")
        self._backup_manager = BackupManager(self._backup_dir)

    def repack(
        self,
        modified_files: list[ModifiedFile],
        papgt_path: str,
        create_backup: bool = True,
        verify_after: bool = True,
        preserve_timestamps: bool = True,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> RepackResult:
        """Execute the full repack pipeline.

        Args:
            modified_files: List of files to repack.
            papgt_path: Path to the PAPGT root index.
            create_backup: Whether to create backups before modifying.
            verify_after: Whether to verify checksums after repacking.
            preserve_timestamps: Whether to preserve file timestamps.
            progress_callback: Optional callback(percentage, message).

        Returns:
            RepackResult with operation details.
        """
        errors = []
        backup_dir_used = ""

        def report(pct: int, msg: str):
            logger.info("[%d%%] %s", pct, msg)
            if progress_callback:
                progress_callback(pct, msg)

        groups = {}
        for mf in modified_files:
            key = mf.package_group
            if key not in groups:
                groups[key] = []
            groups[key].append(mf)

        total_steps = len(groups) * 5 + 2
        step = 0

        if create_backup:
            report(0, "Creating backup of original files...")
            files_to_backup = set()
            files_to_backup.add(papgt_path)
            for group_key, group_files in groups.items():
                pamt_path = group_files[0].pamt_data.path
                files_to_backup.add(pamt_path)
                for mf in group_files:
                    files_to_backup.add(mf.entry.paz_file)

            backup_record = self._backup_manager.create_backup(
                list(files_to_backup),
                description=f"Repack {sum(len(g) for g in groups.values())} files"
            )
            backup_dir_used = backup_record.backup_dir

        papgt_data = parse_papgt(papgt_path)
        logger.info("[REPACK] Working with PAPGT index: %s", papgt_path)
        papgt_raw = bytearray(papgt_data.raw_data)

        last_papgt_crc = 0
        last_pamt_crc = 0
        last_paz_crc = 0

        for group_key, group_files in groups.items():
            pamt_data = group_files[0].pamt_data
            pamt_raw = bytearray(pamt_data.raw_data)
            space_map = build_space_map(pamt_data.file_entries)

            paz_files_modified = {}

            for mf in group_files:
                step += 1
                pct = int((step / total_steps) * 100)
                report(pct, f"Processing {os.path.basename(mf.entry.path)}...")

                processed_data = mf.data

                if mf.entry.compressed and mf.entry.compression_type != 0:
                    processed_data = compress(processed_data, mf.entry.compression_type)

                if mf.entry.encrypted:
                    basename = os.path.basename(mf.entry.path)
                    processed_data = encrypt(processed_data, basename)

                new_comp_size = len(processed_data)
                new_orig_size = len(mf.data)

                paz_path = mf.entry.paz_file
                logger.info("[REPACK] Writing payload for %s: %d bytes (Original source: %d bytes)", 
                            mf.entry.path, new_comp_size, new_orig_size)
                
                new_offset, _ = write_entry_payload(
                    mf.entry,
                    processed_data,
                    space_map,
                    preserve_timestamps=preserve_timestamps,
                )

                paz_files_modified[paz_path] = True

                update_pamt_file_entry(
                    pamt_raw,
                    mf.entry,
                    new_comp_size,
                    new_orig_size,
                    new_offset=new_offset,
                )
                mf.entry.offset = new_offset
                mf.entry.comp_size = new_comp_size
                mf.entry.orig_size = new_orig_size
                logger.info(
                    "Updated file entry: %s offset=0x%08X->0x%08X comp=%d->%d orig=%d->%d",
                    mf.entry.path,
                    mf.entry.offset,
                    new_offset,
                    mf.entry.comp_size, new_comp_size,
                    mf.entry.orig_size, new_orig_size,
                )

            step += 1
            pct = int((step / total_steps) * 100)
            report(pct, f"Computing PAZ checksums for group {group_key}...")

            for paz_path in paz_files_modified:
                with open(paz_path, "rb") as f:
                    paz_data = f.read()
                new_paz_crc = pa_checksum(paz_data)
                new_paz_size = len(paz_data)
                last_paz_crc = new_paz_crc

                paz_basename = os.path.basename(paz_path)
                paz_num = int(os.path.splitext(paz_basename)[0])
                pamt_stem = int(os.path.splitext(os.path.basename(pamt_data.path))[0])
                paz_index = paz_num - pamt_stem

                for table_entry in pamt_data.paz_table:
                    if table_entry.index == paz_index:
                        update_pamt_paz_entry(pamt_raw, table_entry, new_paz_crc, new_paz_size)
                        logger.info(
                            "Updated PAZ[%d] CRC=0x%08X size=%d",
                            paz_index, new_paz_crc, new_paz_size,
                        )
                        break

            step += 1
            pct = int((step / total_steps) * 100)
            report(pct, f"Updating PAMT self-CRC for group {group_key}...")

            new_pamt_crc = update_pamt_self_crc(pamt_raw)
            last_pamt_crc = new_pamt_crc

            pamt_path = pamt_data.path
            logger.info("[REPACK] Writing updated PAMT: %s (New CRC: 0x%08X)", pamt_path, new_pamt_crc)
            
            if preserve_timestamps:
                ts = get_file_timestamps(pamt_path)
            atomic_write(pamt_path, bytes(pamt_raw))
            if preserve_timestamps:
                set_file_timestamps(pamt_path, ts["modified"], ts["accessed"])

            logger.info("PAMT %s self-CRC updated: 0x%08X", group_key, new_pamt_crc)

            step += 1
            pct = int((step / total_steps) * 100)
            report(pct, f"Updating PAPGT entry for group {group_key}...")

            folder_number = int(group_key)
            pamt_crc_offset = get_pamt_crc_offset(papgt_data, folder_number)
            update_papgt_pamt_crc(papgt_raw, pamt_crc_offset, new_pamt_crc)

        step += 1
        pct = int((step / total_steps) * 100)
        report(pct, "Updating PAPGT self-CRC...")

        new_papgt_crc = update_papgt_self_crc(papgt_raw)
        last_papgt_crc = new_papgt_crc

        if preserve_timestamps:
            ts = get_file_timestamps(papgt_path)
        atomic_write(papgt_path, bytes(papgt_raw))
        if preserve_timestamps:
            set_file_timestamps(papgt_path, ts["modified"], ts["accessed"])

        logger.info("PAPGT self-CRC updated: 0x%08X", new_papgt_crc)

        if verify_after:
            step += 1
            report(95, "Verifying checksums...")
            try:
                self._verify_chain(papgt_path, groups)
            except Exception as e:
                errors.append(f"Verification failed: {e}")
                logger.error("Post-repack verification failed: %s", e)

        report(100, "Repack complete!")

        return RepackResult(
            success=len(errors) == 0,
            files_repacked=len(modified_files),
            paz_crc=last_paz_crc,
            pamt_crc=last_pamt_crc,
            papgt_crc=last_papgt_crc,
            backup_dir=backup_dir_used,
            errors=errors,
        )

    def _verify_chain(self, papgt_path: str, groups: dict) -> None:
        """Verify the complete checksum chain after repack."""
        from core.checksum_engine import verify_papgt_checksum, verify_pamt_checksum

        ok, stored, computed = verify_papgt_checksum(papgt_path)
        if not ok:
            raise ValueError(
                f"PAPGT checksum verification failed: "
                f"stored=0x{stored:08X} computed=0x{computed:08X}"
            )

        for group_key, group_files in groups.items():
            pamt_path = group_files[0].pamt_data.path
            ok, stored, computed = verify_pamt_checksum(pamt_path)
            if not ok:
                raise ValueError(
                    f"PAMT {group_key} checksum verification failed: "
                    f"stored=0x{stored:08X} computed=0x{computed:08X}"
                )

        logger.info("All checksums verified successfully")

    def restore_backup(self, backup_dir: str) -> list[str]:
        """Restore files from a backup."""
        return self._backup_manager.restore_backup(backup_dir)

    def list_backups(self) -> list[dict]:
        """List available backups."""
        return self._backup_manager.list_backups()
