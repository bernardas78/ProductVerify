$src_folder = "C:\IsKnown_Images_IsVisible\Train\"
$dest_folder = "C:\IsKnown_Images_IsVisible\Train10\"

Get-ChildItem $src_folder |
ForEach {
    $src_bc_folder = $src_folder + $_.Name
    $dest_bc_folder = $dest_folder + $_.Name

    #echo $bc_folder
    if (-Not (Test-Path -Path $dest_bc_folder)){
        New-Item -Path $dest_bc_folder -ItemType Directory
        }
    Get-ChildItem $src_bc_folder -File | Select-Object -last 10 |
    ForEach {
        copy $_.FullName $dest_bc_folder
    }
}