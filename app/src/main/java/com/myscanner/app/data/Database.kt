package com.myscanner.app.data

import android.content.Context
import androidx.room.ColumnInfo
import androidx.room.Dao
import androidx.room.Database
import androidx.room.Embedded
import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.Insert
import androidx.room.PrimaryKey
import androidx.room.Query
import androidx.room.Relation
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.Transaction
import androidx.room.Update
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase
import kotlinx.coroutines.flow.Flow

/** One saved scan (a "document"), made of one or more pages. */
@Entity(tableName = "documents")
data class DocumentEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val title: String,
    val createdAt: Long,
    val updatedAt: Long,
)

/** One page image inside a document. `position` is its order, starting at 0. */
@Entity(
    tableName = "pages",
    foreignKeys = [
        ForeignKey(
            entity = DocumentEntity::class,
            parentColumns = ["id"],
            childColumns = ["documentId"],
            onDelete = ForeignKey.CASCADE,
        )
    ],
    indices = [Index("documentId")],
)
data class PageEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val documentId: Long,
    val position: Int,
    val imagePath: String,
    /** Text found by OCR. Null until the user runs "Extract text". */
    val ocrText: String? = null,
    /** The original photo, kept so the page can be edited again (null for older or imported pages). */
    @ColumnInfo(defaultValue = "NULL") val sourcePath: String? = null,
    /** Page corners on the original photo, as "x,y;x,y;x,y;x,y" (0..1). */
    @ColumnInfo(defaultValue = "NULL") val quad: String? = null,
    /** Filter name used for this page. */
    @ColumnInfo(defaultValue = "NULL") val filter: String? = null,
    /** Extra clockwise rotation applied after straightening: 0, 90, 180, 270. */
    @ColumnInfo(defaultValue = "0") val rotation: Int = 0,
    /** Drawings on the page (pen, highlighter, signature, text), in the app's own text form. */
    @ColumnInfo(defaultValue = "NULL") val marks: String? = null,
)

data class DocumentWithPages(
    @Embedded val document: DocumentEntity,
    @Relation(parentColumn = "id", entityColumn = "documentId")
    val pagesUnsorted: List<PageEntity>,
)

/** The document's pages in their saved order. */
val DocumentWithPages.pages: List<PageEntity>
    get() = pagesUnsorted.sortedBy { it.position }

@Dao
interface DocumentDao {
    @Transaction
    @Query("SELECT * FROM documents ORDER BY updatedAt DESC")
    fun observeAll(): Flow<List<DocumentWithPages>>

    /** Finds documents whose title OR extracted text contains the search words. */
    @Transaction
    @Query(
        """
        SELECT * FROM documents
        WHERE title LIKE '%' || :query || '%'
           OR id IN (SELECT documentId FROM pages WHERE ocrText LIKE '%' || :query || '%')
        ORDER BY updatedAt DESC
        """
    )
    fun search(query: String): Flow<List<DocumentWithPages>>

    @Transaction
    @Query("SELECT * FROM documents WHERE id = :id")
    fun observeDocument(id: Long): Flow<DocumentWithPages?>

    @Transaction
    @Query("SELECT * FROM documents WHERE id = :id")
    suspend fun getDocument(id: Long): DocumentWithPages?

    @Query("SELECT * FROM pages WHERE id = :id")
    suspend fun getPage(id: Long): PageEntity?

    @Insert
    suspend fun insertDocument(document: DocumentEntity): Long

    @Insert
    suspend fun insertPages(pages: List<PageEntity>)

    @Update
    suspend fun updatePage(page: PageEntity)

    @Update
    suspend fun updatePages(pages: List<PageEntity>)

    @Query("UPDATE documents SET title = :title, updatedAt = :time WHERE id = :id")
    suspend fun rename(id: Long, title: String, time: Long)

    @Query("UPDATE documents SET updatedAt = :time WHERE id = :id")
    suspend fun touch(id: Long, time: Long)

    @Query("DELETE FROM documents WHERE id = :id")
    suspend fun deleteDocument(id: Long)

    @Query("DELETE FROM pages WHERE id = :id")
    suspend fun deletePage(id: Long)

    @Query("DELETE FROM pages WHERE documentId = :documentId")
    suspend fun deletePagesOf(documentId: Long)

    // Small, targeted changes. (Writing back a whole page row that was read earlier could undo a
    // change made in between, e.g. text recognition finishing after the user moved a page.)

    @Query("UPDATE pages SET ocrText = :text WHERE id = :id")
    suspend fun setOcrText(id: Long, text: String?)

    @Query("UPDATE pages SET position = :position WHERE id = :id")
    suspend fun setPosition(id: Long, position: Int)

    @Query("UPDATE pages SET imagePath = :imagePath, rotation = :rotation, marks = :marks WHERE id = :id")
    suspend fun setImage(id: Long, imagePath: String, rotation: Int, marks: String?)

    @Transaction
    suspend fun setPositions(orderedIds: List<Long>) {
        orderedIds.forEachIndexed { i, id -> setPosition(id, i) }
    }

    @Query("SELECT * FROM pages")
    suspend fun allPages(): List<PageEntity>

    @Query("UPDATE pages SET sourcePath = NULL")
    suspend fun forgetSources()

    @Query("SELECT COUNT(*) FROM documents")
    suspend fun documentCount(): Int
}

@Database(entities = [DocumentEntity::class, PageEntity::class], version = 3, exportSchema = false)
abstract class AppDatabase : RoomDatabase() {
    abstract fun documentDao(): DocumentDao

    companion object {
        /** Version 6.2: pages remember their original photo, corners, filter and rotation (for editing). */
        private val MIGRATION_1_2 = object : Migration(1, 2) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE pages ADD COLUMN sourcePath TEXT DEFAULT NULL")
                db.execSQL("ALTER TABLE pages ADD COLUMN quad TEXT DEFAULT NULL")
                db.execSQL("ALTER TABLE pages ADD COLUMN filter TEXT DEFAULT NULL")
                db.execSQL("ALTER TABLE pages ADD COLUMN rotation INTEGER NOT NULL DEFAULT 0")
            }
        }

        /** Version 8: pages remember the drawings made on them. */
        private val MIGRATION_2_3 = object : Migration(2, 3) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE pages ADD COLUMN marks TEXT DEFAULT NULL")
            }
        }

        fun create(context: Context): AppDatabase =
            Room.databaseBuilder(context, AppDatabase::class.java, "scanner.db")
                .addMigrations(MIGRATION_1_2, MIGRATION_2_3)
                .build()
    }
}
