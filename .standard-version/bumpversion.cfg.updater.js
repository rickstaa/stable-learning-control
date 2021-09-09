/**
 * Updater used to update Bumpversion configuration file (i.e. `.bumvpersion.cfg`).
 */

/**
 * Reads the current version from the Bumvpersion configuration file.
 */
module.exports.readVersion = function (contents) {
  // console.debug(`file contents:\n\n${contents}`);
  const version = contents.match(/(?<=current_version = )\d.\d.\d/g)[0];
  // console.debug("found version:", version);
  return version;
};

/**
 * Writes the new version to the Bumvpersion configuration file.
 */
module.exports.writeVersion = function (contents, version) {
  // console.debug(`file contents:\n\n${contents}`);
  return contents.replace(/(?<=current_version = )\d.\d.\d/g, () => {
    // console.debug("replace version with", version);
    return version;
  });
};
