#!/usr/bin/env python3
"""Generate WiX source for the PyInstaller one-folder Windows bundle."""

from __future__ import annotations

import argparse
import hashlib
import uuid
from pathlib import Path
from xml.sax.saxutils import escape


PRODUCT_NAME = "Black Hole Benchmark"
MANUFACTURER = "Physics Sandbox"
UPGRADE_CODE = "7f7550b6-7d63-4c16-8dc6-4d6a03bbda51"
REGISTRY_ROOT = r"Software\PhysicsSandbox\BlackHoleBenchmark"
COMPONENT_GUID_NAMESPACE = uuid.UUID("1d70d2da-493c-4c46-aa2c-12f49a315c17")


def xml_attr(value: str) -> str:
    return escape(value, {'"': "&quot;"})


def wix_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def component_guid(value: str) -> str:
    return str(uuid.uuid5(COMPONENT_GUID_NAMESPACE, value)).upper()


class DirectoryNode:
    def __init__(self, name: str, directory_id: str) -> None:
        self.name = name
        self.directory_id = directory_id
        self.children: dict[str, DirectoryNode] = {}
        self.files: list[Path] = []


def add_file(root: DirectoryNode, dist_dir: Path, file_path: Path) -> None:
    rel_path = file_path.relative_to(dist_dir)
    node = root
    current_parts: list[str] = []
    for part in rel_path.parts[:-1]:
        current_parts.append(part)
        node = node.children.setdefault(part, DirectoryNode(part, wix_id("dir", "/".join(current_parts))))
    node.files.append(file_path)


def emit_directory(node: DirectoryNode, dist_dir: Path, lines: list[str], depth: int) -> None:
    indent = "  " * depth
    if node.directory_id == "INSTALLFOLDER":
        lines.append(f'{indent}<Directory Id="INSTALLFOLDER" Name="{xml_attr(PRODUCT_NAME)}">')
    else:
        lines.append(f'{indent}<Directory Id="{node.directory_id}" Name="{xml_attr(node.name)}">')

    for file_path in sorted(node.files, key=lambda item: item.name.lower()):
        rel_path = file_path.relative_to(dist_dir).as_posix()
        component_id = wix_id("cmp", rel_path)
        registry_name = wix_id("file", rel_path)
        source_path = str(file_path)
        lines.append(f'{indent}  <Component Id="{component_id}" Guid="{component_guid(rel_path)}">')
        lines.append(f'{indent}    <File Source="{xml_attr(source_path)}" />')
        lines.append(
            f'{indent}    <RegistryValue Root="HKCU" Key="{xml_attr(REGISTRY_ROOT)}\\InstalledFiles" '
            f'Name="{registry_name}" Type="integer" Value="1" KeyPath="yes" />'
        )
        lines.append(f"{indent}  </Component>")

    for child in sorted(node.children.values(), key=lambda item: item.name.lower()):
        emit_directory(child, dist_dir, lines, depth + 1)

    lines.append(f"{indent}</Directory>")


def emit_component_refs(root: DirectoryNode, dist_dir: Path, lines: list[str]) -> None:
    for file_path in sorted(dist_dir.rglob("*")):
        if file_path.is_file():
            rel_path = file_path.relative_to(dist_dir).as_posix()
            lines.append(f'      <ComponentRef Id="{wix_id("cmp", rel_path)}" />')


def generate_wix(dist_dir: Path) -> str:
    exe_path = dist_dir / "BlackHoleBenchmark.exe"
    if not exe_path.exists():
        raise SystemExit(f"Expected PyInstaller executable is missing: {exe_path}")

    files = sorted(path for path in dist_dir.rglob("*") if path.is_file())
    if not files:
        raise SystemExit(f"No files found in PyInstaller output: {dist_dir}")

    root = DirectoryNode(PRODUCT_NAME, "INSTALLFOLDER")
    for file_path in files:
        add_file(root, dist_dir, file_path.resolve())

    lines: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<Wix xmlns="http://wixtoolset.org/schemas/v4/wxs" xmlns:ui="http://wixtoolset.org/schemas/v4/wxs/ui">',
        (
            f'  <Package Name="{xml_attr(PRODUCT_NAME)}" Manufacturer="{xml_attr(MANUFACTURER)}" '
            'Version="$(var.ProductVersion)" UpgradeCode="{' + UPGRADE_CODE + '}" Scope="perUser">'
        ),
        '    <SummaryInformation Description="Relativistic black hole raytracer and performance benchmark" />',
        '    <MajorUpgrade DowngradeErrorMessage="A newer version of Black Hole Benchmark is already installed." />',
        '    <MediaTemplate EmbedCab="yes" />',
        '    <Icon Id="AppIcon" SourceFile="$(var.IconPath)" />',
        '    <Property Id="ARPPRODUCTICON" Value="AppIcon" />',
        '    <Launch Condition="VersionNT64" Message="Black Hole Benchmark requires 64-bit Windows." />',
        (
            '    <Launch Condition="VersionNT &gt;= 1000" '
            'Message="Black Hole Benchmark requires Windows 10 or later. Windows 11 is recommended." />'
        ),
        '',
        '    <StandardDirectory Id="LocalAppDataFolder">',
    ]
    emit_directory(root, dist_dir.resolve(), lines, 3)
    lines.extend(
        [
            "    </StandardDirectory>",
            "",
            '    <StandardDirectory Id="ProgramMenuFolder">',
            f'      <Directory Id="ApplicationProgramsFolder" Name="{xml_attr(PRODUCT_NAME)}">',
            (
                '        <Component Id="StartMenuShortcutComponent" '
                f'Guid="{component_guid("shortcut/start-menu")}">'
            ),
            (
                f'          <Shortcut Id="StartMenuShortcut" Name="{xml_attr(PRODUCT_NAME)}" '
                'Description="Run the relativistic black hole benchmark" '
                'Target="[INSTALLFOLDER]BlackHoleBenchmark.exe" WorkingDirectory="INSTALLFOLDER" Icon="AppIcon" />'
            ),
            '          <RemoveFolder Id="ApplicationProgramsFolder" On="uninstall" />',
            (
                f'          <RegistryValue Root="HKCU" Key="{xml_attr(REGISTRY_ROOT)}" '
                'Name="StartMenuShortcut" Type="integer" Value="1" KeyPath="yes" />'
            ),
            "        </Component>",
            "      </Directory>",
            "    </StandardDirectory>",
            "",
            '    <StandardDirectory Id="DesktopFolder">',
            (
                '      <Component Id="DesktopShortcutComponent" '
                f'Guid="{component_guid("shortcut/desktop")}">'
            ),
            (
                f'        <Shortcut Id="DesktopShortcut" Name="{xml_attr(PRODUCT_NAME)}" '
                'Description="Run the relativistic black hole benchmark" '
                'Target="[INSTALLFOLDER]BlackHoleBenchmark.exe" WorkingDirectory="INSTALLFOLDER" Icon="AppIcon" />'
            ),
            (
                f'        <RegistryValue Root="HKCU" Key="{xml_attr(REGISTRY_ROOT)}" '
                'Name="DesktopShortcut" Type="integer" Value="1" KeyPath="yes" />'
            ),
            "      </Component>",
            "    </StandardDirectory>",
            "",
            '    <Feature Id="MainFeature" Title="Black Hole Benchmark" Level="1">',
        ]
    )
    emit_component_refs(root, dist_dir.resolve(), lines)
    lines.extend(
        [
            '      <ComponentRef Id="StartMenuShortcutComponent" />',
            '      <ComponentRef Id="DesktopShortcutComponent" />',
            "    </Feature>",
            "",
            '    <ui:WixUI Id="WixUI_InstallDir" InstallDirectory="INSTALLFOLDER" />',
            "  </Package>",
            "</Wix>",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", required=True, type=Path, help="PyInstaller one-folder output directory.")
    parser.add_argument("--output", required=True, type=Path, help="Generated WiX source path.")
    args = parser.parse_args()

    dist_dir = args.dist_dir.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generate_wix(dist_dir), encoding="utf-8")


if __name__ == "__main__":
    main()
